"""Language-model backends: a local llama.cpp model, and a scripted one for tests.

Every call is greedy (temperature 0) and returns, besides the text, the log-probability of each
generated token and of the most likely alternatives at that position. Those numbers are the
model's own uncertainty, used as verification signals (see signals.py).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TokenInfo:
    token: str
    logprob: float
    top: dict[str, float] = field(default_factory=dict)   # alternative token -> logprob


@dataclass(frozen=True)
class Generation:
    text: str
    tokens: tuple[TokenInfo, ...] = ()
    prompt_tokens: int = 0


Message = dict[str, str]


class Backend(Protocol):
    name: str

    def chat(self, messages: Sequence[Message], max_tokens: int,
             top_logprobs: int = 5) -> Generation: ...


class ScriptedBackend:
    """Returns pre-written answers in order (tests and dry runs). Logprobs are given as-is."""

    def __init__(self, answers: Sequence[str | Generation], name: str = "scripted") -> None:
        self.answers = list(answers)
        self.name = name
        self.calls: list[list[Message]] = []

    def chat(self, messages, max_tokens, top_logprobs=5) -> Generation:
        self.calls.append(list(messages))
        if not self.answers:
            raise RuntimeError("scripted backend has no answer left")
        a = self.answers.pop(0)
        return a if isinstance(a, Generation) else Generation(a)


class LlamaBackend:
    """A GGUF model run on CPU with llama-cpp-python; greedy decoding, fixed seed."""

    def __init__(self, model_path: str | Path, n_ctx: int = 4096, n_threads: int | None = None,
                 seed: int = 0) -> None:
        from llama_cpp import Llama  # imported here: optional dependency

        self.path = Path(model_path)
        self.name = self.path.stem
        self.llm = Llama(model_path=str(self.path), n_ctx=n_ctx, n_threads=n_threads,
                         seed=seed, logits_all=True, verbose=False)

    def chat(self, messages, max_tokens, top_logprobs=5) -> Generation:
        out = self.llm.create_chat_completion(
            messages=list(messages), max_tokens=max_tokens, temperature=0.0, top_k=1,
            logprobs=True, top_logprobs=top_logprobs,
        )
        choice = out["choices"][0]
        tokens = []
        for t in (choice.get("logprobs") or {}).get("content") or []:
            tokens.append(TokenInfo(t["token"], float(t["logprob"]),
                                    {a["token"]: float(a["logprob"])
                                     for a in t.get("top_logprobs") or []}))
        return Generation(choice["message"]["content"] or "", tuple(tokens),
                          int(out.get("usage", {}).get("prompt_tokens", 0)))


def binary_probability(tokens: Sequence[TokenInfo], yes: set[str], no: set[str]) -> float | None:
    """P(yes) / (P(yes) + P(no)) at the first generated token that is a yes/no word.

    Tokens are compared after stripping spaces and quotes and lower-casing. Returns None when no
    generated token is a yes/no word or neither appears among that position's alternatives.
    """
    def norm(tok: str) -> str:
        return tok.strip().strip('"').lower()

    for t in tokens:
        if norm(t.token) in yes | no:
            cands = dict(t.top)
            cands.setdefault(t.token, t.logprob)
            p_yes = sum(math.exp(lp) for tok, lp in cands.items() if norm(tok) in yes)
            p_no = sum(math.exp(lp) for tok, lp in cands.items() if norm(tok) in no)
            return p_yes / (p_yes + p_no) if p_yes + p_no > 0 else None
    return None
