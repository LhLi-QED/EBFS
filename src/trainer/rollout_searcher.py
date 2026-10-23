import re
import math
import json
import torch
import sys

from typing import List, Optional, Union
from copy import deepcopy
from scipy.signal import find_peaks
from lean.verifier import verify_lean4_file
from lean.proof import ProofSummarizer

USER_PROMPT = """Complete the following Lean 4 code:
```lean4
{striped_header}

{striped_theorem}
  sorry
```
Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
"""

ASSISTANT_RESUME_ROOT = """{global_cot}```lean4
"""


ASSISTANT_RESUME_NODE = """{global_cot}```lean4
{code_before}"""

ROOT_ROLLOUT_OUTPUT = """{global_cot}```lean4
{code_before}```"""

ROLLOUT_CHOSEN_COT = """{global_cot}```lean4
{code_before}"""

class Candidate:
    def __init__(
        self, 
        segments: List[dict],
        peaks: int,
        ancestor: Union["Candidate", "SearchHandler"], 
        is_solved: bool,
        meta_data: Optional[dict]
    ):
        self.segments = segments
        self.meta_data = meta_data
        self.peaks = peaks
        self.ancestor = ancestor
        self.handler = ancestor.handler if isinstance(ancestor, Candidate) else ancestor
        self.depth = self.ancestor.depth + 1

        self.succeed = []
        self.n_failed_expansion = 0
        self.is_solved = is_solved

        if self.is_solved:
            self.backward_solution(self.full_code)

    @property
    def children_peaks(self):
      return sum([c.peaks for c in self.succeed])

    @property
    def goal(self):
      return self.segments[-1]["goal"]

    @property
    def full_code(self):
      return _get_full_code(self.segments)

    @property
    def prompt(self):
        messages = [
            {"role": "user", "content": USER_PROMPT.format(striped_header=self.handler.header, striped_theorem=self.handler.formal_statement)},
            {"role": "assistant", "content": ASSISTANT_RESUME_NODE.format(code_before=self.full_code, global_cot=self.handler.global_cot)}
            ]
        prompt = self.handler.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False).strip()
        if self.handler.tokenizer.eos_token and prompt.endswith(self.handler.tokenizer.eos_token):
            prompt = prompt[:-len(self.handler.tokenizer.eos_token)]
        return prompt

    @property
    def score(self):
        return (1 + math.log(1 + self.children_peaks)) / (len(self.succeed) + self.n_failed_expansion + 1e-5)

    def backward_solution(self, full_code):
        if isinstance(self.ancestor, Candidate):
            self.ancestor.is_solved = True
            self.ancestor.backward_solution(full_code)
        else:
            self.ancestor.full_code = full_code

    def to_json(self):
        dict = {
            "segments": self.segments,
            "peaks":self.peaks,
            "depth":self.depth,
            "n_failed_expansion": self.n_failed_expansion,
            "is_solved": self.is_solved,
            "meta_data":self.meta_data,
            "succeed": [s.to_json() for s in self.succeed]
        }
        return dict

    @classmethod
    def build_from_verifier(
        cls, 
        ancestor: Union["Candidate", "SearchHandler"], 
        generated_text: str,
        generated_code: str,
        segments: List[dict], 
        entropy_info: List[dict],
        tokenizer,
        is_solved: bool
    ):
            
        entropy = [item["entropy"] for item in entropy_info]
        token_ids = [item["token_id"] for item in entropy_info]

        code_token_ids = tokenizer.encode(generated_code.lstrip("\n").rstrip("\n"), add_special_tokens=False)
        code_match_start, code_match_end = _match_token_ids(token_ids, code_token_ids, 0)
        entropy = entropy[code_match_start: code_match_end + 1]
        token_ids = token_ids[code_match_start: code_match_end + 1]
        handler = ancestor.handler if isinstance(ancestor, Candidate) else ancestor

        seg_start = 0
        if isinstance(ancestor, Candidate):
            is_match, seg_start = _match_segments(ancestor.segments, segments)
            if is_match:
                wpg_ancestor = ancestor
            else:
                wpg_ancestor = handler
        else:
            wpg_ancestor = handler
            seg_start = 0
            is_match = True
        if seg_start >= len(segments) or not is_match:
            if isinstance(wpg_ancestor, Candidate):
                wpg_ancestor.n_failed_expansion += 1
            return None


        if len(segments) == 0 and is_solved:
            candidate = cls(
                            segments=segments,
                            peaks=0,
                            ancestor=wpg_ancestor, 
                            is_solved=is_solved,
                            meta_data={"generated_text": generated_text, "generated_code": generated_code, "all_code_valid": generated_code, "step_code_valid": generated_code,"rejected": None}
                        )
            wpg_ancestor.succeed.append(candidate)
            handler.update_candidates(candidate)
            return
        all_generated_code_valid = _get_full_code(segments[seg_start:])
        first_invalid_step = _find_first_invalid_step(all_generated_code_valid, generated_code)
        if is_match and not is_solved and len(first_invalid_step.strip()) > 0:
            rejected = first_invalid_step
        else:
            rejected = None

        self_peaks = 0
        child_peaks = 0
        start = 0
        for k in range(seg_start, len(segments)):
            segment = segments[k]
            code = segment['tactic_code']
            seg_token_ids = tokenizer.encode(code.lstrip("\n").rstrip("\n"), add_special_tokens=False)
            match_start, match_end = _match_token_ids(token_ids, seg_token_ids, start)
            if match_end > 0:
                start = match_end
            peaks, _ = find_peaks(entropy[match_start: match_end], height=0.3)
            child_peaks = len(peaks)
            if len(segments[seg_start:k]) > 0:
                candidate = cls(
                    segments=segments[:k],
                    peaks=self_peaks,
                    ancestor=wpg_ancestor, 
                    is_solved=is_solved,
                    meta_data={"generated_text": generated_text, "generated_code": generated_code, "all_code_valid": all_generated_code_valid, "step_code_valid": _get_full_code(segments[seg_start:k]), "rejected": rejected}
                )
                self_peaks = child_peaks
                wpg_ancestor.succeed.append(candidate)
                handler.update_candidates(candidate)
                wpg_ancestor = candidate
        if len(segments[seg_start:]) > 0:
            candidate = cls(
                segments=segments,
                peaks=self_peaks,
                ancestor=wpg_ancestor, 
                is_solved=is_solved,
                meta_data={"generated_text": generated_text, "generated_code": generated_code, "all_code_valid": all_generated_code_valid, "step_code_valid": all_generated_code_valid, "rejected": rejected}
            )
            wpg_ancestor.succeed.append(candidate)
            handler.update_candidates(candidate)



class SearchHandler:
    def __init__(self, id, header, formal_statement, tokenizer, search_budget, **kwargs):
        self.id = id.strip()
        self.header = header.strip()
        self.formal_statement = formal_statement.strip()
        self.search_budget = search_budget
        self.candidates = []
        self.succeed = []
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.depth = 0

        self.is_solved = False
        self.messages = [
            {"role": "user", "content": USER_PROMPT.format(striped_header=self.header, striped_theorem=self.formal_statement)},
        ]
        self.prompt = tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)

        data = {"header": self.header + "\n\n", "formal_statement": ""}
        self.proof_summarizer = ProofSummarizer(data=data, scheduler=None)
        self.global_cot = None
        self.goal = None
        self.full_code = None

        self.last_selected = self


    @property
    def handler(self):
        return self

    def select(self, n):
        self.search_budget -= n
        if len(self.succeed) > 0:
            self.candidates.sort(key=lambda x: x.score, reverse=True)
            self.last_selected = self.candidates[0]
            return self.candidates[0]
        else:
            self.last_selected = self
            return self

    def rollout(self):
        if self.global_cot is None or self.global_cot.strip() == "":
            return []
        solution_path = []
        for c in self.succeed:
            if c.is_solved:
                solution_path = [c]
                while len(c.succeed) > 0 and any(s.is_solved for s in c.succeed):
                    if len(c.succeed) > 1 or (len(c.succeed) == 1 and c.meta_data["generated_text"] != c.succeed[0].meta_data["generated_text"]):
                        for succeed in c.succeed:
                            if succeed.is_solved:
                                solution_path.append(succeed)
                                c = succeed
                                break
                    else:
                        c = c.succeed[0]
                break
        rollouts = []
        if len(solution_path) > 1:
            rollouts = self.rollout_candidates(solution_path)
        else:
            rollouts.extend(self.rollout_root())
        # removed_shortcut_rollouts = []
        # for data in rollouts:
        #     if "apply?" not in data["chosen"] and "exact?" not in data["chosen"]:
        #         removed_shortcut_rollouts.append(data)
        # return removed_shortcut_rollouts
        return rollouts

    def rollout_candidates(self, solution_path):
        eos_token = self.tokenizer.eos_token
        rollouts = []
        for k, c in enumerate(solution_path):
            a = c.ancestor
            prompt = a.prompt
            if isinstance(a, SearchHandler):
                assert k == 0
                prompt = USER_PROMPT.format(striped_header=self.header, striped_theorem=self.formal_statement)
                messages = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                next_candidate_ancestor = solution_path[1].ancestor
                if len(next_candidate_ancestor.meta_data["step_code_valid"].strip()) > 0:
                    chosen_code = next_candidate_ancestor.meta_data["step_code_valid"]
                    chosen = ROLLOUT_CHOSEN_COT.format(global_cot=self.global_cot, code_before=chosen_code)
                    if c.meta_data["rejected"] is not None:
                        rollouts.append({
                            "input": prompt, 
                            "chosen": chosen, 
                            "neutral": c.meta_data["all_code_valid"][len(chosen_code): ],
                            "rejected": c.meta_data["rejected"]
                            })
                    else:
                        rollouts.append({
                            "input": prompt, 
                            "chosen": chosen, 
                            })
            elif c != solution_path[-1]:
                next_candidate_ancestor = solution_path[k+1].ancestor
                valid_steps = next_candidate_ancestor.meta_data["step_code_valid"]
                if len((valid_steps).strip()) > 0:
                    if c.meta_data["rejected"] is not None:
                        rollouts.append({
                            "input": prompt, 
                            "chosen": valid_steps, 
                            "neutral": c.meta_data["all_code_valid"][len(valid_steps): ],
                            "rejected": c.meta_data["rejected"]
                            })
                    else:
                        rollouts.append({
                            "input": prompt,
                            "chosen": valid_steps,
                            })
            else:
                all_generated_text = c.meta_data["generated_text"]
                if len(all_generated_text.strip()) > 0:
                    if not all_generated_text.endswith(eos_token):
                        all_generated_text += eos_token
                    rollouts.append({"input": prompt,"chosen": all_generated_text})
        return rollouts

        
    def rollout_root(self):
        chosen_node = None
        eos_token = self.tokenizer.eos_token
        for c in self.succeed:
            if c.is_solved:
                all_succeed = []
                cur = c
                all_succeed.append(cur)
                while len(cur.succeed) > 0:
                    all_succeed.append(cur.succeed[0])
                    cur = cur.succeed[0]
                all_succeed.append(cur)
                
                if all(len(s.succeed) <= 1 for s in all_succeed):
                    chosen_node = all_succeed[-1]
                    break
        if chosen_node is not None:
            prompt = USER_PROMPT.format(striped_header=self.header, striped_theorem=self.formal_statement)
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            output = ROOT_ROLLOUT_OUTPUT.format(global_cot=self.global_cot, code_before=chosen_node.full_code) + eos_token
            return [{"input": prompt,"chosen": output}]
        else:
            return []

    def is_terminal(self):
        return self.search_budget <= 0 or self.is_solved

    def update_candidates(self, candidate: "Candidate"):
        self.candidates.append(candidate)
        self.is_solved = self.is_solved or candidate.is_solved

    def update(self, vllm_outputs):
        selected_candidate = self.last_selected
        previous_code = _get_full_code(selected_candidate.segments) if isinstance(selected_candidate, Candidate) else ""
        all_text, all_entropy_info = _parse_vllm_outputs(vllm_outputs, self.tokenizer)
        all_proof = []
        verify_results = []

        if isinstance(selected_candidate, SearchHandler) and self.global_cot is None:
            cot, code = _extract_generated_code_init(all_text[0], self.tokenizer.eos_token)
            self.global_cot = cot
            if len(self.messages) == 1:
                self.messages.append({"role": "assistant", "content": ASSISTANT_RESUME_ROOT.format(global_cot=self.global_cot)})
            else:
                self.messages[1]["content"] = ASSISTANT_RESUME_ROOT.format(global_cot=self.global_cot)
            prompt = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=False).strip()
            if self.tokenizer.eos_token and prompt.endswith(self.tokenizer.eos_token):
                prompt = prompt[:-len(self.tokenizer.eos_token)]
            self.prompt = prompt
            all_code = [code]
            all_text = [all_text[0]]
            all_entropy_info = [all_entropy_info[0]]
            if code == "":
                return
        else:
            all_code = [_extract_generated_code(text, self.tokenizer.eos_token) for text in all_text]
        for code in all_code:
            p = self.proof_summarizer.analyze(previous_code + code, require_verification=False)
            all_proof.append(p)

        for p in all_proof:
            results = verify_lean4_file(code=p.full_code, ast=True, tactics=True, timeout=600)
            verify_results.append(results)
        for text, entropy_info, results, code, proof in zip(all_text, all_entropy_info, verify_results, all_code, all_proof):
            if len(results.get("sorries", ["REPL FAILED!"])) == 0:
                if self.goal is None:
                    try:
                        self.goal = results["ast"]["tactics"][0]["stateBefore"]
                    except:
                        continue
                self.is_solved = self.is_solved or results.get("complete", False)
                Candidate.build_from_verifier(
                    selected_candidate,
                    text, 
                    code,
                    proof.segmentation(result=results),
                    entropy_info,
                    self.tokenizer,
                    results["complete"]
                )
                
    def to_json(self):
        results = {
            "id": self.id, 
            "header": self.header, 
            "formal_statement": self.formal_statement,
            "global_cot": self.global_cot,
            "full_code": self.full_code, 
            "succeed": [c.to_json() for c in self.succeed],
            }
        return results
        
def _remove_prefix(prefix, text):
    assert prefix in text
    return text[len(prefix):]

def _parse_vllm_outputs(vllm_outputs, tokenizer):
    all_text = []
    all_entropy_info = []
    if not isinstance(vllm_outputs, list):
        # outputs = model.completions.create(model="./models/Qwen", prompt="Hello!", n=4, logprobs=0, max_tokens=512, temperature=0.6)
        choices = vllm_outputs.choices
        for completion in choices:
            all_text.append(completion.text)
            top_logprobs = completion.logprobs.top_logprobs
            token_ids = tokenizer.encode(completion.text)
            info = []
            for logprobs, token_id in zip(top_logprobs, token_ids):
                logprobs = list(logprobs.values())
                entropy = _compute_entropy(logprobs)
                info.append({"token_id": token_id, "entropy": entropy})
            all_entropy_info.append(info)
    else:
        # sampling_params = SamplingParams(n=4, top_p=1.0, temperature=0.6, max_tokens=2048, logprobs=0)
        for completion in vllm_outputs[0].outputs:
            all_text.append(completion.text)
            top_logprobs = completion.logprobs
            token_ids = completion.token_ids
            info = []
            for logprob, token_id in zip(top_logprobs, token_ids):
                logprobs = [p.logprob for p in list(logprob.values())]
                entropy = _compute_entropy(logprobs)
                info.append({"token_id": token_id, "entropy": entropy})
            all_entropy_info.append(info)

    return all_text, all_entropy_info

def _compute_entropy(logits):
    logits = torch.tensor(logits)
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy.item()


def _extract_generated_code(output, eos_token):
    output = output.replace(eos_token, "")
    output = output.replace("```", "").lstrip("\n")
    if not output.endswith("\n"):
        output = output + "\n"
    return output

def _extract_generated_code_init(output, eos_token):

    output = output.replace(eos_token, "").strip()
    spans = output.split("```lean4")
    if len(spans) > 1:
        cot = "```lean4".join(spans[:-1])
        try:
            code = spans[-1].replace("```", "").lstrip("\n")
            if not code.endswith("\n"):
                code = code + "\n"
        except:
            print(spans[-1])
            raise 
    else:
        cot = output
        code = ""
    return cot, code

def _match_token_ids(token_ids, segment_token_ids, match_start):
    j = 0
    match_end = match_start
    while j < len(segment_token_ids)  and match_start < len(token_ids):
        if segment_token_ids[j] == token_ids[match_start]:
            match_end = match_start
            while match_end < len(token_ids) and j < len(segment_token_ids) and segment_token_ids[j] == token_ids[match_end]:
                match_end += 1
                j += 1
            if j == len(segment_token_ids):
                return match_start, match_end
            else:
              match_start = match_end
              j = 0
        else:
            match_start += 1
    if j == len(segment_token_ids):
        return match_start, match_end
    else:
        return -1, -1

def _match_segments(ancestor_segments, segments):
    k = 0
    for segment in ancestor_segments:
        if k == len(segments):
            return True, k
        if segment["goal"] == segments[k]["goal"]:
            k+=1
        else:
            return False, 0
    return True, k

def _get_full_code(segments):
    return "".join(segment['tactic_code'] for segment in segments)

def _find_first_invalid_step(valid_code, all_generated_code):
    if len(all_generated_code) == 0 or len(valid_code) >= len(all_generated_code):
        return ""
    res = all_generated_code[len(valid_code): ]
    lines = res.splitlines()
    ret = []
    for line in lines:
        if line.strip() == "" or line.strip().startswith("--") or line.strip().startswith("· --") or line.strip().startswith("have"):
            ret.append(line)
            continue
        else:
            ret.append(line)
            if _check_brackets_balance("\n".join(ret)):
                break
    return "\n".join(ret)

def _check_brackets_balance(s):
    stack = []
    bracket_pairs = {')': '(', ']': '['}
    
    for char in s:
        if char in ['(', '[']:
            stack.append(char)
        elif char in [')', ']']:
            if not stack or stack[-1] != bracket_pairs[char]:
                return False
            stack.pop()
    return len(stack) == 0

def _process_rejected(prompt: str, rejected: str, chosen: str):
    if len(chosen) == 0:
        return prompt, rejected
    _cr_prefix = ""
    i = 0
    for i in range(len(rejected)):
        if i < len(chosen) and rejected[i] == chosen[i]:
            _cr_prefix += rejected[i]
        else:
            break
    prompt = prompt + _cr_prefix
    rejected = rejected[i:]
    return prompt, rejected

def _verify_step(search_handler, selected_candidate, vllm_outputs):
    search_handler.update(vllm_outputs, selected_candidate)
    return search_handler
