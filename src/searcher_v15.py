import math
from typing import List, Optional, Union

from lean.verifier import verify_lean4_file
from lean.proof import ProofSummarizer
from searcher import (
    compute_entropy,
    count_entropy_peaks,
    match_token_ids,
    parse_vllm_outputs,
    _match_segments,
)

_PROMPT_PREFIX = (
    "Complete the following Lean 4 code with explanatory comments"
    " preceding each line of code:\n\n```lean4\n"
)


def _build_init_prompt(header: str, formal_statement: str) -> str:
    """Build the initial completion prompt (first round, no prior code)."""
    return f"{_PROMPT_PREFIX}{header}\n\n{formal_statement}\n"


def _build_resume_prompt(
    header: str,
    formal_statement: str,
    code: str,
    state_comment: str,
) -> str:
    """Build a resume prompt for tree search expansion.

    Appends the tactic state comment block after the verified code prefix
    so the model can see the current proof state before generating next tactics.
    """
    return (
        f"{_PROMPT_PREFIX}{header}\n\n{formal_statement}\n"
        f"{code}{state_comment}"
    )


def _extract_code(output: str, eos_token: str) -> str:
    """Extract code from model output, stripping eos and closing ```."""
    output = output.replace(eos_token, "")
    idx = output.find("```")
    if idx >= 0:
        output = output[:idx]
    return output.lstrip("\n")


class V15Candidate:
    def __init__(
        self,
        segments: List[dict],
        peaks: int,
        ancestor: Union["V15Candidate", "V15SearchHandler"],
        is_solved: bool,
        meta_data: Optional[dict] = None,
    ):
        self.segments = segments
        self.peaks = peaks
        self.ancestor = ancestor
        self.meta_data = meta_data or {}
        self.handler = ancestor.handler if isinstance(ancestor, V15Candidate) else ancestor
        self.depth = ancestor.depth + 1

        self.succeed: List["V15Candidate"] = []
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
    def state_comment(self) -> str:
        """The tactic state comment block from the last segment."""
        if self.segments:
            return self.segments[-1].get("state_comment", "")
        return ""

    @property
    def full_code(self) -> str:
        return "".join(seg["tactic_code"] for seg in self.segments)

    @property
    def prompt(self) -> str:
        """Build the completion prompt for expanding this node."""
        return _build_resume_prompt(
            header=self.handler.header,
            formal_statement=self.handler.formal_statement,
            code=self.full_code,
            state_comment=self.state_comment,
        )

    @property
    def score(self) -> float:
        """UCB-like score for candidate selection (entropy-based)."""
        return (1 + math.log(1 + self.children_peaks)) / (
            len(self.succeed) + self.n_failed_expansion + 1e-5
        )

    def _propagate_solution(self, full_code: str):
        if isinstance(self.ancestor, V15Candidate):
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
        ancestor: Union["V15Candidate", "V15SearchHandler"],
        generated_text: str,
        generated_code: str,
        segments: List[dict],
        entropy_info: List[dict],
        tokenizer,
        is_solved: bool,
    ):
        """Build candidate chain from verified proof attempt."""
        handler = ancestor.handler if isinstance(ancestor, V15Candidate) else ancestor

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

        # Determine where new segments start
        seg_start = 0
        is_match = True
        if isinstance(ancestor, V15Candidate):
            is_match, seg_start = _match_segments(ancestor.segments, segments)
            if not is_match:
                seg_start = 0
                ancestor = handler

        if seg_start >= len(segments) or not is_match:
            if isinstance(ancestor, V15Candidate):
                ancestor.n_failed_expansion += 1
            return None

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

        # Build candidate chain
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


# ---------------------------------------------------------------------------
# V15SearchHandler — manages proof search for one theorem
# ---------------------------------------------------------------------------

class V15SearchHandler:
    """Proof search handler for DeepSeek-Prover-V1.5 CoT mode.

    Differences from SearchHandler:
      - Completion-style prompt (no chat template)
      - No global CoT extraction — reasoning is inline in code
      - All outputs processed equally on every round
      - Resume prompt includes tactic state comment
    """

    def __init__(
        self,
        id: str,
        header: str,
        formal_statement: str,
        tokenizer,
        search_budget: int,
        **kwargs,
    ):
        self.id = id.strip()
        self.header = header.strip()
        self.formal_statement = formal_statement.strip()
        self.search_budget = search_budget
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.depth = 0

        self.candidates: List[V15Candidate] = []
        self.succeed: List[V15Candidate] = []
        self.is_solved = False
        self.goal: Optional[str] = None
        self.full_code: str = ""

        data = {"header": self.header + "\n\n", "formal_statement": formal_statement}
        self.proof_summarizer = ProofSummarizer(data=data, scheduler=None)
        self.last_selected = self

    @property
    def handler(self):
        return self

    @property
    def prompt(self) -> str:
        """Initial completion prompt (root node)."""
        return _build_init_prompt(self.header, self.formal_statement)

    def select(self, n: int) -> Union["V15Candidate", "V15SearchHandler"]:
        """Select best candidate for next expansion."""
        self.search_budget -= n
        if self.candidates:
            self.candidates.sort(key=lambda x: x.score, reverse=True)
            self.last_selected = self.candidates[0]
            return self.candidates[0]
        self.last_selected = self
        return self

    def is_terminal(self) -> bool:
        return self.search_budget <= 0 or self.is_solved

    def update_candidates(self, candidate: V15Candidate):
        self.candidates.append(candidate)
        self.is_solved = self.is_solved or candidate.is_solved

    def update(self, vllm_outputs):
        """Process vLLM outputs: extract code, verify, segment, build candidates."""
        selected = self.last_selected
        previous_code = selected.full_code if isinstance(selected, V15Candidate) else ""
        all_text, all_entropy_info = parse_vllm_outputs(vllm_outputs, self.tokenizer)

        # Extract code from all outputs (no special first-round handling)
        all_code = [_extract_code(t, self.tokenizer.eos_token) for t in all_text]

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
            V15Candidate.build_from_verifier(
                ancestor=selected,
                generated_text=text,
                generated_code=code,
                segments=p.segmentation(result=results),
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
            "full_code": self.full_code,
            "succeed": [c.to_json() for c in self.succeed],
        }
