import re
import math
import json
import torch
import traceback
from typing import List, Optional, Union
from scipy.signal import find_peaks

from lean.verifier import verify_lean4_file
from lean.proof import ProofSummarizer


_DEEPSEEK_V2_USER = """Complete the following Lean 4 code:
```lean4
{header}

{formal_statement}
  sorry
```
Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""

_DEEPSEEK_V2_ASSISTANT_RESUME = """{global_cot}
```lean4
{code_before}"""


class PromptManager:

    _TEMPLATES = {
        "deepseek-v2":   (_DEEPSEEK_V2_USER,  _DEEPSEEK_V2_ASSISTANT_RESUME,  None),
    }

    def __init__(self, prover: str = "deepseek-v2"):
        if prover not in self._TEMPLATES:
            raise ValueError(
                f"Unknown prover '{prover}'. "
                f"Available: {list(self._TEMPLATES.keys())}"
            )
        self.prover = prover
        self._user_tpl, self._assistant_tpl, self._system = self._TEMPLATES[prover]

    @property
    def system_prompt(self) -> Optional[str]:
        return self._system

    @property
    def has_assistant_resume(self) -> bool:
        return self._assistant_tpl is not None

    def init_prompt(self, header: str, formal_statement: str) -> str:
        """First-round prompt (before any CoT or code is generated)."""
        return self._user_tpl.format(
            header=header.strip(),
            formal_statement=formal_statement.strip(),
        )

    def resume_prompt(self, formal_statement: str, code_before: str, global_cot: str) -> str:
        return self._assistant_tpl.format(
            global_cot=global_cot,
            code_before=code_before
        )

    def build_chat(self, header: str, formal_statement: str,
                   assistant_content: Optional[str] = None) -> list:
        """Build chat message list according to prover type.

        Args:
            header: Lean 4 file header (imports, options, opens).
            formal_statement: Theorem statement with docstring.
            assistant_content: If not None, append as an assistant message
                               (for resume / continuation prompts).

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        chat = []
        if self._system is not None:
            chat.append({"role": "system", "content": self._system})
        chat.append({"role": "user", "content": self.init_prompt(header, formal_statement)})
        if assistant_content is not None:
            chat.append({"role": "assistant", "content": assistant_content})
        return chat

_CODE_BLOCK_RE = re.compile(r"```lean4\n(.*?)```", re.DOTALL)


def extract_code_init(output: str, eos_token: str) -> tuple:
    """Extract (cot, code) from the first-round LLM output.

    For CoT provers, the output contains natural-language reasoning
    followed by a ```lean4 code block.  We split on the *last*
    ```lean4 block so that earlier blocks in the CoT are preserved.
    """
    output = output.replace(eos_token, "").strip()
    spans = output.split("```lean4")
    if len(spans) > 1:
        cot = "```lean4".join(spans[:-1])
        code = spans[-1].replace("```", "").lstrip("\n")
    else:
        cot = output
        code = ""
    return cot, code


def extract_code(output: str, eos_token: str) -> str:
    """Extract code from continuation-generation output (no CoT split)."""
    return output.replace(eos_token, "").replace("```", "").lstrip("\n")

def compute_entropy(logprobs) -> float:
    """Shannon entropy from a list of log-probabilities."""
    logits = torch.tensor(logprobs)
    pd = torch.nn.functional.softmax(logits, dim=-1)
    return (torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)).item()

def count_entropy_peaks(entropies: list, height: float = 0.3) -> int:
    """Count peaks in an entropy sequence."""
    if not entropies:
        return 0
    peaks, _ = find_peaks(entropies, height=height)
    return len(peaks)

def match_token_ids(token_ids: list, seg_token_ids: list, start: int = 0):
    """Find the span [match_start, match_end) in token_ids that matches
    seg_token_ids, searching from `start`.  Returns (-1, -1) on failure."""
    j = 0
    ms = start
    while j < len(seg_token_ids) and ms < len(token_ids):
        if seg_token_ids[j] == token_ids[ms]:
            me = ms
            while me < len(token_ids) and j < len(seg_token_ids) and seg_token_ids[j] == token_ids[me]:
                me += 1
                j += 1
            if j == len(seg_token_ids):
                return ms, me
            ms = me
            j = 0
        else:
            ms += 1
    return (ms, me) if j == len(seg_token_ids) else (-1, -1)

def post_process_segments(segments):
    for i, seg in enumerate(segments):
        if "exact?" in seg["tactic_code"] or "apply?" in seg["tactic_code"]:
            return segments[:i]
    return segments


class Candidate:
    def __init__(
        self,
        segments: List[dict],
        peaks: int,
        ancestor: Union["Candidate", "SearchHandler"],
        is_solved: bool,
        meta_data: Optional[dict] = None,
    ):
        self.segments = segments
        self.peaks = peaks
        self.ancestor = ancestor
        self.meta_data = meta_data or {}
        self.handler = ancestor.handler if isinstance(ancestor, Candidate) else ancestor
        self.depth = ancestor.depth + 1

        self.succeed: List["Candidate"] = []
        self.n_failed_expansion = 0
        self.is_solved = is_solved

        if self.is_solved:
            self._propagate_solution(self.full_code)

    @property
    def children_peaks(self) -> int:
        return sum(c.peaks for c in self.succeed)

    @property
    def goal(self) -> str:
        return self.segments[-1]["goal"] if self.segments else ""

    @property
    def full_code(self) -> str:
        return "".join(seg["tactic_code"] for seg in self.segments)

    @property
    def prompt(self) -> str:
        pm = self.handler.prompt_manager
        tokenizer = self.handler.tokenizer
        assistant_content = pm.resume_prompt(
            self.handler.formal_statement, self.full_code, self.handler.global_cot or ""
        )
        chat = pm.build_chat(
            self.handler.header, self.handler.formal_statement,
            assistant_content=assistant_content,
        )
        inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).strip()
        if tokenizer.bos_token and inputs.startswith(tokenizer.bos_token):
            inputs = inputs[len(tokenizer.bos_token):]
        if tokenizer.eos_token and inputs.endswith(tokenizer.eos_token):
            inputs = inputs[:-len(tokenizer.eos_token)]
        return inputs

    @property
    def score(self) -> float:
        """UCB-like score for candidate selection."""
        return (1 + math.log(1 + self.children_peaks)) / (
            len(self.succeed) + self.n_failed_expansion + 1e-5
        )

    def _propagate_solution(self, full_code: str):
        """Backward-propagate the solved flag to ancestors."""
        if isinstance(self.ancestor, Candidate):
            self.ancestor.is_solved = True
            self.ancestor._propagate_solution(full_code)
        else:
            self.ancestor.full_code = full_code

    def to_json(self) -> dict:
        return {
            "segments": self.segments,
            "peaks": self.peaks,
            "depth": self.depth,
            "n_failed_expansion": self.n_failed_expansion,
            "is_solved": self.is_solved,
            "meta_data": self.meta_data,
            "succeed": [s.to_json() for s in self.succeed],
        }

    # -- factory --

    @classmethod
    def build_from_verifier(
        cls,
        ancestor: Union["Candidate", "SearchHandler"],
        generated_text: str,
        generated_code: str,
        segments: List[dict],
        entropy_info: List[dict],
        tokenizer,
        is_solved: bool,
    ):
        """Build one or more Candidate nodes from a verified proof attempt.

        This method:
          1. Aligns token-level entropy with tactic segments
          2. Counts entropy peaks per segment
          3. Creates a chain of Candidate nodes (one per new segment)
        """
        handler = ancestor.handler if isinstance(ancestor, Candidate) else ancestor

        # Align entropy to code tokens
        all_entropy = [e["entropy"] for e in entropy_info]
        all_token_ids = [e["token_id"] for e in entropy_info]
        code_token_ids = tokenizer.encode(
            generated_code.lstrip("\n").rstrip("\n"), add_special_tokens=False
        )
        code_start, code_end = match_token_ids(all_token_ids, code_token_ids, 0)
        if code_start >= 0:
            all_entropy = all_entropy[code_start:code_end]
            all_token_ids = all_token_ids[code_start:code_end]

        # Determine where new segments start (skip ancestor's segments)
        seg_start = 0
        is_match = True
        if isinstance(ancestor, Candidate):
            is_match, seg_start = _match_segments(ancestor.segments, segments)
            if not is_match:
                seg_start = 0
                ancestor = handler  # fall back to root

        if seg_start >= len(segments) or not is_match:
            if isinstance(ancestor, Candidate):
                ancestor.n_failed_expansion += 1
            return None

        # Handle trivial solved case (no segments)
        if not segments and is_solved:
            c = cls(
                segments=[],
                peaks=0,
                ancestor=ancestor,
                is_solved=True,
                meta_data={"generated_text": generated_text, "generated_code": generated_code},
            )
            ancestor.succeed.append(c)
            handler.update_candidates(c)
            return

        # Build candidate chain: one node per new segment
        all_valid_code = "".join(seg["tactic_code"] for seg in segments[seg_start:])
        search_pos = 0
        current_ancestor = ancestor
        current_peaks = 0

        for k in range(seg_start, len(segments)):
            seg = segments[k]
            seg_token_ids = tokenizer.encode(
                seg["tactic_code"].lstrip("\n").rstrip("\n"), add_special_tokens=False
            )
            ms, me = match_token_ids(all_token_ids, seg_token_ids, search_pos)
            if me > 0:
                search_pos = me
            seg_peaks = count_entropy_peaks(all_entropy[ms:me]) if ms >= 0 else 0

            if k > seg_start:
                c = cls(
                    segments=segments[:k],
                    peaks=current_peaks,
                    ancestor=current_ancestor,
                    is_solved=is_solved,
                    meta_data={
                        "generated_text": generated_text,
                        "generated_code": generated_code,
                        "valid_code": "".join(s["tactic_code"] for s in segments[seg_start:k]),
                    },
                )
                current_ancestor.succeed.append(c)
                handler.update_candidates(c)
                current_ancestor = c

            current_peaks = seg_peaks

        if segments[seg_start:]:
            c = cls(
                segments=segments,
                peaks=current_peaks,
                ancestor=current_ancestor,
                is_solved=is_solved,
                meta_data={
                    "generated_text": generated_text,
                    "generated_code": generated_code,
                    "valid_code": all_valid_code,
                },
            )
            current_ancestor.succeed.append(c)
            handler.update_candidates(c)


class SearchHandler:
    """Manages the proof search tree for a single theorem.

    Lifecycle:
      1. Create with theorem info and search budget
      2. Call select(n) to pick the best candidate for expansion
      3. Call update(vllm_outputs, selected) to process LLM outputs
      4. Repeat until is_terminal()
    """

    def __init__(
        self,
        id: str,
        header: str,
        formal_statement: str,
        tokenizer,
        search_budget: int,
        prover: str = "deepseek-v2",
        **kwargs,
    ):
        self.id = id.strip()
        self.header = header.strip()
        self.formal_statement = formal_statement.strip()
        self.search_budget = search_budget
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.depth = 0

        self.prompt_manager = PromptManager(prover)
        self.candidates: List[Candidate] = []
        self.succeed: List[Candidate] = []
        self.is_solved = False
        self.global_cot: Optional[str] = None
        self.goal: Optional[str] = None
        self.full_code: str = ""

        data = {"header": self.header + "\n\n", "formal_statement": ""}
        self.proof_summarizer = ProofSummarizer(data=data, scheduler=None)
        self.last_selected = self

    @property
    def handler(self):
        return self
    
    @property
    def prompt(self) -> str:
        pm = self.prompt_manager
        tokenizer = self.tokenizer
        if self.global_cot is not None:
            assistant_content = pm.resume_prompt(
                self.handler.formal_statement, "", self.handler.global_cot
            )
            chat = pm.build_chat(
                self.handler.header, self.handler.formal_statement,
                assistant_content=assistant_content,
            )
            inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).strip()
        else:
            chat = pm.build_chat(
                self.handler.header, self.handler.formal_statement,
                assistant_content=None,
            )
            inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if tokenizer.bos_token and inputs.startswith(tokenizer.bos_token):
            inputs = inputs[len(tokenizer.bos_token):]
        if tokenizer.eos_token and inputs.endswith(tokenizer.eos_token):
            inputs = inputs[:-len(tokenizer.eos_token)]
        return inputs

    def select(self, n: int) -> Union["Candidate", "SearchHandler"]:
        """Select the best candidate for the next expansion round."""
        self.search_budget -= n
        if self.succeed:
            self.candidates.sort(key=lambda x: x.score, reverse=True)
            self.last_selected = self.candidates[0]
            return self.candidates[0]
        self.last_selected = self
        return self

    def is_terminal(self) -> bool:
        return self.search_budget <= 0 or self.is_solved

    def update_candidates(self, candidate: Candidate):
        self.candidates.append(candidate)
        self.is_solved = self.is_solved or candidate.is_solved

    def update(self, vllm_outputs):
        """Process vLLM outputs: parse, verify, segment, build candidates."""
        selected = self.last_selected
        previous_code = selected.full_code if isinstance(selected, Candidate) else ""
        all_text, all_entropy_info = parse_vllm_outputs(vllm_outputs, self.tokenizer)

        # First round with CoT: extract CoT from the first output
        if isinstance(selected, SearchHandler) and self.global_cot is None:
            cot, code = extract_code_init(all_text[0], self.tokenizer.eos_token)
            self.global_cot = cot
            all_text = [all_text[0]]
            all_entropy_info = [all_entropy_info[0]]
            all_code = [code]
            if not code:
                return
        else:
            all_code = [extract_code(t, self.tokenizer.eos_token) for t in all_text]

        # Verify and segment each output
        for text, entropy_info, code in zip(all_text, all_entropy_info, all_code):
            if not code.strip():
                continue

            p = self.proof_summarizer.analyze(
                previous_code + code, require_verification=False
            )
            results = verify_lean4_file(
                code=p.full_code, ast=True, tactics=True, timeout=2000
            )

            if results.get("sorries", ["REPL FAILED!"]):
                continue

            if self.goal is None:
                try:
                    self.goal = results["ast"]["tactics"][0]["stateBefore"]
                except (KeyError, IndexError):
                    pass

            is_complete = results.get("complete", False) and "exact?" not in code and "apply?" not in code
            Candidate.build_from_verifier(
                ancestor=selected,
                generated_text=text,
                generated_code=code,
                segments=post_process_segments(p.segmentation(result=results)),
                entropy_info=entropy_info,
                tokenizer=self.tokenizer,
                is_solved=is_complete,
            )

        return {
            "complete": self.is_solved,
            "error": None,
        }

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "header": self.header,
            "formal_statement": self.formal_statement,
            "global_cot": self.global_cot,
            "full_code": self.full_code,
            "succeed": [c.to_json() for c in self.succeed],
        }


def parse_vllm_outputs(vllm_outputs, tokenizer):
    all_text = []
    all_entropy_info = []

    if not isinstance(vllm_outputs, list):
        # OpenAI-compatible API
        for choice in vllm_outputs.choices:
            all_text.append(choice.text)
            info = []
            if choice.logprobs and choice.logprobs.top_logprobs:
                token_ids = tokenizer.encode(choice.text)
                for logprobs_dict, tid in zip(choice.logprobs.top_logprobs, token_ids):
                    lps = list(logprobs_dict.values())
                    info.append({"token_id": tid, "entropy": compute_entropy(lps)})
            all_entropy_info.append(info)
    else:
        # Native vLLM
        for completion in vllm_outputs[0].outputs:
            all_text.append(completion.text)
            info = []
            if completion.logprobs:
                for logprob_dict, tid in zip(completion.logprobs, completion.token_ids):
                    lps = [p.logprob for p in logprob_dict.values()]
                    info.append({"token_id": int(tid), "entropy": compute_entropy(lps)})
            all_entropy_info.append(info)

    return all_text, all_entropy_info


def _match_segments(ancestor_segments, segments):
    """Check if segments starts with ancestor_segments.
    Returns (is_match, seg_start_index)."""
    k = 0
    for seg in ancestor_segments:
        if k >= len(segments):
            return True, k
        if seg["goal"] == segments[k]["goal"]:
            k += 1
        else:
            return False, 0
    return True, k
