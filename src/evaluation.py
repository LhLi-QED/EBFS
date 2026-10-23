"""
Evaluation pipeline for Lean 4 proof tree search.

Uses OpenAI-compatible API (e.g., vLLM serving endpoints) for LLM inference,
with ray.remote for parallel Lean verification.  All vLLM instances share a
single task queue for flexible load balancing.

Two-level budget:
  - proof_budget: max number of independent search attempts per theorem
  - step_budget:  max number of LLM calls per search attempt

Usage:
    - Initialize vLLM endpoint with:
        ```bash
        HOST_IP=$(hostname -I | awk '{print $1}')
        export VLLM_HOST_IP=$HOST_IP
        vllm serve ../models/DeepSeek-Prover-V2-7B \
            --host $HOST_IP \
            --port 7536 \
            --trust-remote-code \
            --tensor-parallel-size 1 \
            --pipeline-parallel-size 1  \
            --data-parallel-size 8 \
            --gpu-memory-utilization 0.95 \
            --served-model-name Prover \
            --max-model-len 16384 \
            --no-enable-prefix-caching
        ```

    - Run the evaluation pipeline with:
        ```bash
        python evaluation.py \
            --llm_endpoints "10.0.0.1:7536,10.0.0.2:7536,10.0.0.3:7536" \
            --data_path ../data/test.json \
            --tokenizer_path ../models/DeepSeek-Prover-V2-7B \
            --model Prover \
            --benchmark proverbench \
            --save_dir ../results/ds2/proverbench \
            --proof_budget 32 \
            --step_budget 16 \
            --prover deepseek-v2 \
            --llm_concurrency 16
        ```
"""

import os
import sys
import json
import time
import random
import asyncio
import logging
import argparse
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import ray
from openai import AsyncOpenAI
from transformers import AutoTokenizer, AutoConfig

from searcher import (
    SearchHandler,
    Candidate,
    PromptManager,
    extract_code_init,
    extract_code
)
from searcher_v15 import (
    V15SearchHandler,
    V15Candidate,
    _extract_code as v15_extract_code,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

@dataclass
class TaskItem:
    """A pending LLM generation task."""
    name: str
    search_handler: object  # SearchHandler | V15SearchHandler
    selected: object  # Candidate | V15Candidate | SearchHandler | V15SearchHandler
    proof_attempt: int  # which proof attempt this belongs to


@dataclass
class TheoremState:
    """Tracks the state of a single theorem across all proof attempts."""
    name: str
    header: str
    formal_statement: str
    proof_budget_remaining: int
    is_solved: bool = False
    active_searches: int = 0  # number of in-flight search handlers
    best_proof: Optional[str] = None
    total_steps_used: int = 0
    proof_attempts_used: int = 0


@dataclass
class VerifyTask:
    """A pending Lean verification task."""
    name: str
    search_handler: object  # SearchHandler | V15SearchHandler
    selected: object
    llm_response: object
    proof_attempt: int


# ============================================================
# Ray remote: Lean verification + segmentation
# ============================================================

@ray.remote(num_cpus=8)
def verify_and_segment(
    search_handler,
    llm_response,
) -> dict:
    try:
        return search_handler.update(llm_response), search_handler
    except:
        logger.error(f"Verification failed: {traceback.format_exc()}")
        return {"complete": search_handler.is_solved, "error": traceback.format_exc()}, search_handler


# ============================================================
# LLM Client Pool
# ============================================================

class LLMPool:
    """Manages a pool of OpenAI-compatible LLM endpoints.

    Each endpoint is a vLLM instance serving at http://{ip}:{port}/v1.
    Tasks are dispatched from a shared asyncio.Queue.
    """

    def __init__(
        self,
        endpoints: List[str],
        model: str = "Prover",
        tokenizer = None,
        max_model_len: int = 16384,
        api_key: str = "EMPTY",
        temperature: float = 0.6,
        top_logprobs: int = 20,
        llm_concurrency: int = 8,
    ):
        self.endpoints = endpoints
        self.model = model
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_logprobs = top_logprobs

        # Create one AsyncOpenAI client per endpoint
        self.clients: List[AsyncOpenAI] = []
        self.llm_concurrency = llm_concurrency
        for ep in endpoints:
            if not ep.startswith("http"):
                ep = f"http://{ep}"
            if not ep.endswith("/v1"):
                ep = f"{ep}/v1"
            self.clients.append(
                AsyncOpenAI(base_url=ep, api_key=api_key)
            )

        # Semaphore per client: each vLLM instance handles num llm_concurrency requests at a time
        self.semaphores = [asyncio.Semaphore(llm_concurrency) for _ in self.clients]

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """Send a completion request to the first available endpoint."""
        # Find an available client
        while True:
            for idx, sem in enumerate(self.semaphores):
                if sem.locked():
                    continue
                async with sem:
                    return await self._call(idx, prompt, system_prompt)
            # All busy, wait a bit
            await asyncio.sleep(0.05)

    async def _call(
        self, client_idx: int, prompt: str, system_prompt: Optional[str]
    ) -> dict:
        """Make the actual API call."""
        client = self.clients[client_idx]

        # Use completions API (not chat) for prefix-continuation style
        try:
            response = await client.completions.create(
                model=self.model,
                prompt=prompt,
                n=1,
                temperature=self.temperature,
                max_tokens=self.max_model_len - len(self.tokenizer.encode(prompt)) - 8,
                logprobs=self.top_logprobs,
                stop=None,
                timeout=3600
            )
        except Exception as e:
            logger.error(f"LLM call failed on endpoint {self.endpoints[client_idx]}: {e}")
            raise

        return {
            "text": response.choices[0].text,
            "llm_response": response,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }


# ============================================================
# Core evaluation loop
# ============================================================

class Evaluator:
    """Orchestrates the evaluation pipeline.

    Architecture:
      - Shared task queue (asyncio.Queue) of TaskItems
      - N worker coroutines, each grabs a task, calls LLM, submits verification
      - Verification results are collected and fed back into search handlers
    """

    def __init__(
        self,
        llm_pool: LLMPool,
        tokenizer,
        dataset: List[dict],
        prover: str,
        proof_budget: int,
        step_budget: int,
        save_dir: str,
        max_concurrent_verify: int = 64,
    ):
        self.llm_pool = llm_pool
        self.tokenizer = tokenizer
        self.prover = prover
        self.proof_budget = proof_budget
        self.step_budget = step_budget
        self.save_dir = save_dir
        self.max_concurrent_verify = max_concurrent_verify

        # Per-theorem state
        self.theorems: Dict[str, TheoremState] = {}
        for item in dataset:
            name = item["name"].strip()
            self.theorems[name] = TheoremState(
                name=name,
                header=item["header"],
                formal_statement=item["formal_statement"],
                proof_budget_remaining=proof_budget,
            )

        # Task queue
        self.task_queue: asyncio.Queue[TaskItem] = asyncio.Queue()

        # Verification tracking
        self.verify_sem = asyncio.Semaphore(max_concurrent_verify)
        self.pending_verify: List[asyncio.Task] = []

        # Stats
        self.n_solved = 0
        self.n_total = len(dataset)
        self.total_llm_calls = 0
        self.start_time = None

    def _create_search_handler(self, name: str):
        ts = self.theorems[name]
        if self.prover == "deepseek-v1.5":
            return V15SearchHandler(
                id=name,
                header=ts.header,
                formal_statement=ts.formal_statement,
                tokenizer=self.tokenizer,
                search_budget=self.step_budget,
            )
        return SearchHandler(
            id=name,
            header=ts.header,
            formal_statement=ts.formal_statement,
            tokenizer=self.tokenizer,
            search_budget=self.step_budget,
            prover=self.prover,
        )

    def _save_result(self, name: str, handler, attempt: int):
        """Save a solved theorem's result to disk."""
        ts = self.theorems[name]
        result = {
            "name": name,
            "formal_statement": ts.formal_statement,
            "proof": handler.full_code,
            "global_cot": getattr(handler, "global_cot", None),
            "proof_attempts_used": ts.proof_attempts_used,
            "total_steps_used": ts.total_steps_used,
            "proof_tree": handler.to_json(),
        }
        path = os.path.join(self.save_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    async def _enqueue_new_search(self, name: str):
        """Start a new proof attempt for a theorem if budget allows."""
        ts = self.theorems[name]
        if ts.is_solved or ts.proof_budget_remaining <= 0:
            return
        ts.proof_budget_remaining -= 1
        ts.proof_attempts_used += 1
        ts.active_searches += 1

        handler = self._create_search_handler(name)
        task = TaskItem(
            name=name,
            search_handler=handler,
            selected=handler.select(n=1),  # first round: selected is the handler itself
            proof_attempt=ts.proof_attempts_used,
        )
        await self.task_queue.put(task)

    async def _handle_search_step(self, task: TaskItem):
        """Execute one search step: LLM call -> verify -> update tree."""
        handler = task.search_handler
        selected = task.selected
        ts = self.theorems[task.name]

        if ts.is_solved:
            ts.active_searches -= 1
            return

        # Build prompt
        prompt = selected.prompt
        system_prompt = (
            handler.prompt_manager.system_prompt
            if isinstance(handler, SearchHandler) else None
        )
        # LLM call
        try:
            llm_result = await self.llm_pool.generate(prompt, system_prompt)
        except Exception as e:
            logger.error(f"[{task.name}] LLM call failed: {e}")
            ts.active_searches -= 1
            return

        self.total_llm_calls += 1
        ts.total_steps_used += 1

        text = llm_result["text"]

        # Parse output — V1.5 CoT has no global_cot phase
        if isinstance(handler, V15SearchHandler):
            code = v15_extract_code(text, handler.tokenizer.eos_token)
        elif isinstance(selected, SearchHandler) and handler.global_cot is None:
            cot, code = extract_code_init(text, handler.tokenizer.eos_token)
            if not code:
                # CoT extracted but no code yet; continue search
                if not handler.is_terminal():
                    next_selected = handler.select(n=1)
                    next_task = TaskItem(
                        name=task.name,
                        search_handler=handler,
                        selected=next_selected,
                        proof_attempt=task.proof_attempt,
                    )
                    await self.task_queue.put(next_task)
                else:
                    self._handle_terminal(task)
                return
        else:
            code = extract_code(text, handler.tokenizer.eos_token)

        if not code.strip():
            if not handler.is_terminal():
                next_selected = handler.select(n=1)
                next_task = TaskItem(
                    name=task.name,
                    search_handler=handler,
                    selected=next_selected,
                    proof_attempt=task.proof_attempt,
                )
                await self.task_queue.put(next_task)
            else:
                self._handle_terminal(task)
            return

        # Submit verification (non-blocking)
        verify_task = VerifyTask(
            name=task.name,
            search_handler=handler,
            selected=selected,
            llm_response=llm_result["llm_response"],
            proof_attempt=task.proof_attempt,
        )
        atask = asyncio.create_task(self._verify_and_continue(verify_task))
        self.pending_verify.append(atask)

    async def _verify_and_continue(self, vtask: VerifyTask):
        """Submit Lean verification via ray, then continue the search."""
        async with self.verify_sem:
            handler = vtask.search_handler
            ts = self.theorems[vtask.name]

            if ts.is_solved:
                ts.active_searches -= 1
                return

            # Ray remote verification
            ref = verify_and_segment.remote(
                vtask.search_handler,
                vtask.llm_response,
            )

            try:
                vresult, handler = await asyncio.get_event_loop().run_in_executor(
                    None, ray.get, ref
                )
            except Exception as e:
                logger.error(f"[{vtask.name}] Verification failed: {e}")
                self._handle_terminal_from_vtask(vtask)
                return

            if vresult.get("error") is None:
                if handler.is_solved:
                    ts.is_solved = True
                    ts.best_proof = handler.full_code
                    self.n_solved += 1
                    self._save_result(vtask.name, handler, vtask.proof_attempt)
                    elapsed = time.time() - self.start_time
                    logger.info(
                        f"SOLVED [{self.n_solved}/{self.n_total}] {vtask.name} "
                        f"(attempt={vtask.proof_attempt}, "
                        f"steps={ts.total_steps_used}, "
                        f"time={elapsed:.0f}s)"
                    )
                    ts.active_searches -= 1
                    return

            # Continue search or start new attempt
            if not handler.is_terminal():
                next_selected = handler.select(n=1)
                next_task = TaskItem(
                    name=vtask.name,
                    search_handler=handler,
                    selected=next_selected,
                    proof_attempt=vtask.proof_attempt,
                )
                await self.task_queue.put(next_task)
            else:
                self._handle_terminal_from_vtask(vtask)

    def _handle_terminal(self, task: TaskItem):
        """Handle a search handler that has reached its terminal state."""
        ts = self.theorems[task.name]
        ts.active_searches -= 1
        if not ts.is_solved and ts.proof_budget_remaining > 0:
            # Schedule a new proof attempt
            asyncio.create_task(self._enqueue_new_search(task.name))

    def _handle_terminal_from_vtask(self, vtask: VerifyTask):
        """Handle terminal state from a verify task."""
        ts = self.theorems[vtask.name]
        ts.active_searches -= 1
        if not ts.is_solved and ts.proof_budget_remaining > 0:
            asyncio.create_task(self._enqueue_new_search(vtask.name))

    async def _worker(self, worker_id: int):
        """Worker coroutine: pulls tasks from the queue and processes them."""
        while True:
            try:
                task = self.task_queue.get_nowait()
            except asyncio.QueueEmpty:
                # Check if we're done
                if self._all_done():
                    return
                await asyncio.sleep(0.1)
                continue

            ts = self.theorems[task.name]
            if ts.is_solved:
                ts.active_searches -= 1
                self.task_queue.task_done()
                continue

            await self._handle_search_step(task)
            self.task_queue.task_done()

    def _all_done(self) -> bool:
        """Check if all theorems are either solved or exhausted."""
        for ts in self.theorems.values():
            if not ts.is_solved and (ts.proof_budget_remaining > 0 or ts.active_searches > 0):
                return False
        return self.task_queue.empty()

    async def run(self):
        """Main evaluation loop."""
        self.start_time = time.time()
        logger.info(
            f"Starting evaluation: {self.n_total} theorems, "
            f"proof_budget={self.proof_budget}, step_budget={self.step_budget}, "
            f"prover={self.prover}, endpoints={len(self.llm_pool.endpoints)}"
        )

        # Seed initial proof attempts (one per theorem, up to queue capacity)
        initial_batch = min(
            len(self.llm_pool.endpoints) * self.llm_pool.llm_concurrency * 2,
            sum(1 for ts in self.theorems.values() if ts.proof_budget_remaining > 0),
        )
        names = [n for n, ts in self.theorems.items() if ts.proof_budget_remaining > 0]
        if initial_batch // (len(self.llm_pool.endpoints) * self.llm_pool.llm_concurrency) < 1:
            repeat = (len(self.llm_pool.endpoints) * self.llm_pool.llm_concurrency) // len(names) + 1
            for _ in range(repeat):
                for name in names:
                    await self._enqueue_new_search(name)
        else:
            for name in names[:initial_batch]:
                await self._enqueue_new_search(name)
        

        # Launch workers (one per LLM endpoint)
        n_workers = len(self.llm_pool.endpoints) * self.llm_pool.llm_concurrency
        workers = [asyncio.create_task(self._worker(i)) for i in range(n_workers)]

        # Periodically enqueue new searches and log progress
        while not self._all_done():
            await asyncio.sleep(10.0)

            # Enqueue new searches for theorems that need them
            for name, ts in self.theorems.items():
                if (
                    not ts.is_solved
                    and ts.proof_budget_remaining > 0
                    and ts.active_searches == 0
                    and self.task_queue.qsize() < n_workers * 2
                ):
                    await self._enqueue_new_search(name)

            # Log progress
            elapsed = time.time() - self.start_time
            n_exhausted = sum(
                1 for ts in self.theorems.values()
                if not ts.is_solved and ts.proof_budget_remaining <= 0 and ts.active_searches == 0
            )
            logger.info(
                f"[{elapsed:.0f}s] solved={self.n_solved}/{self.n_total}, "
                f"exhausted={n_exhausted}, "
                f"llm_calls={self.total_llm_calls}, "
                f"queue={self.task_queue.qsize()}"
            )

        # Wait for all workers to finish
        for w in workers:
            w.cancel()

        # Wait for pending verifications
        if self.pending_verify:
            await asyncio.gather(*self.pending_verify, return_exceptions=True)

        # Final report
        elapsed = time.time() - self.start_time
        logger.info("=" * 60)
        logger.info(f"Evaluation complete in {elapsed:.1f}s")
        logger.info(f"  Solved: {self.n_solved}/{self.n_total} ({self.n_solved/self.n_total*100:.1f}%)")
        logger.info(f"  Total LLM calls: {self.total_llm_calls}")
        logger.info(f"  Results saved to: {self.save_dir}")

        # Save summary
        summary = {
            "n_total": self.n_total,
            "n_solved": self.n_solved,
            "pass_rate": self.n_solved / self.n_total if self.n_total > 0 else 0,
            "total_llm_calls": self.total_llm_calls,
            "elapsed_seconds": elapsed,
            "proof_budget": self.proof_budget,
            "step_budget": self.step_budget,
            "prover": self.prover,
            "per_theorem": {},
        }
        for name, ts in self.theorems.items():
            summary["per_theorem"][name] = {
                "solved": ts.is_solved,
                "proof_attempts_used": ts.proof_attempts_used,
                "total_steps_used": ts.total_steps_used,
            }
        summary_path = os.path.join(self.save_dir, "_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(data_path: str, benchmark: str, save_dir: str) -> List[dict]:
    """Load dataset and skip already-solved theorems."""
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if isinstance(all_data, dict):
        if benchmark == "all":
            dataset = []
            for bm in all_data:
                dataset.extend(all_data[bm])
        else:
            dataset = all_data[benchmark]
    else:
        dataset = all_data

    # Skip already solved
    finished = set()
    if os.path.exists(save_dir):
        for fname in os.listdir(save_dir):
            if fname.endswith(".json") and not fname.startswith("_"):
                finished.add(fname.rsplit(".", 1)[0])

    remaining = [d for d in dataset if d["name"].strip() not in finished]
    logger.info(
        f"Dataset: {len(dataset)} total, {len(finished)} already solved, "
        f"{len(remaining)} remaining"
    )
    return remaining


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Lean 4 proof tree search evaluation with OpenAI-compatible API"
    )
    # LLM endpoints
    parser.add_argument(
        "--llm_endpoints", type=str, required=True,
        help="Comma-separated list of vLLM instance addresses (ip:port)"
    )
    parser.add_argument("--model", type=str, default="default",
                        help="Model name for the OpenAI API (default: 'default')")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer (for entropy alignment). "
                             "If not set, uses --model as tokenizer path.")

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="putnambench")
    parser.add_argument("--save_dir", type=str, required=True)

    # Budget
    parser.add_argument("--proof_budget", type=int, default=32,
                        help="Max independent search attempts per theorem")
    parser.add_argument("--step_budget", type=int, default=32,
                        help="Max LLM calls per search attempt")

    # Prover
    parser.add_argument("--prover", type=str, default="deepseek-v2",
                        choices=["deepseek-v1.5", "deepseek-v2"])

    # Generation params
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--top_logprobs", type=int, default=20)

    # Concurrency
    parser.add_argument("--max_concurrent_verify", type=int, default=64,
                        help="Max concurrent Lean verification tasks")
    parser.add_argument("--llm_concurrency", type=int, default=8,
                        help="Max concurrent generation tasks per llm endpoint")
    
    args = parser.parse_args()

    # Setup
    os.makedirs(args.save_dir, exist_ok=True)

    if not ray.is_initialized():
        ray.init(runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
            }
        })

    # Parse endpoints
    llms = args.llm_endpoints.split(",")
    # llms.reverse()
    random.shuffle(llms)
    endpoints = [ep.strip() for ep in llms if ep.strip()]
    if not endpoints:
        raise ValueError("No LLM endpoints provided")
    logger.info(f"LLM endpoints: {endpoints}")

    # Tokenizer
    tokenizer_path = args.tokenizer_path or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    max_model_len = args.max_model_len


    # Load dataset
    dataset = load_dataset(args.data_path, args.benchmark, args.save_dir)
    if not dataset:
        logger.info("All theorems already solved. Nothing to do.")
        return


    # Create LLM pool
    llm_pool = LLMPool(
        endpoints=endpoints,
        model=args.model,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        api_key=args.api_key,
        temperature=args.temperature,
        top_logprobs=args.top_logprobs,
        llm_concurrency=args.llm_concurrency,
    )

    # Create evaluator and run
    evaluator = Evaluator(
        llm_pool=llm_pool,
        tokenizer=tokenizer,
        dataset=dataset,
        prover=args.prover,
        proof_budget=args.proof_budget,
        step_budget=args.step_budget,
        save_dir=args.save_dir,
        max_concurrent_verify=args.max_concurrent_verify,
    )

    asyncio.run(evaluator.run())


if __name__ == "__main__":
    main()
