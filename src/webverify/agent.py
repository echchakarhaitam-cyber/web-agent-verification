"""The episode loop: a language model acts on a MiniWoB++ page, then reports success.

One episode:

1. The model sees the instruction, the page (observation.py) and the history of its previous
   actions with any error, and answers with one JSON action (actions.py).
2. Invalid answers are not executed; they cost a step and the parse error is shown back.
3. The loop stops when the model answers ``done``, when the page itself ends the episode (e.g.
   a submit button was pressed), or when the step budget is used up.
4. If the model has not reported yet, it is asked once more for a ``done`` report on the final
   page. Every episode therefore ends with a CLAIM (success true/false and a 0-100 confidence),
   which is compared with the benchmark's reward.
5. An independent JUDGE call, with a fresh prompt and without the agent's history or claim,
   is asked whether the final page shows the task accomplished.

Only the page and the agent's own history reach the model; the reward never does.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from webverify.actions import Done, InvalidAction, parse_action
from webverify.env import MiniWobEnv, Observation
from webverify.llm import Backend, Generation, binary_probability

SYSTEM = (
    "You operate a web page to complete a task. Each page element is listed as "
    '[id] tag "text" (attributes). Answer with exactly one JSON object and nothing else:\n'
    '{"action": "click", "id": <id>}\n'
    '{"action": "type", "id": <id>, "text": "<text>"}   (replaces the content of a text field)\n'
    '{"action": "select", "id": <id>, "option": "<option text>"}   (drop-down lists)\n'
    '{"action": "done", "success": <true|false>, "confidence": <0-100>}\n'
    "Use done when you believe the task is finished or cannot be finished; success says whether "
    "you accomplished it and confidence how sure you are."
)
REPORT = (
    "You can take no more actions. Report now with a done action: did you accomplish the task? "
    'Answer with {"action": "done", "success": <true|false>, "confidence": <0-100>}.'
)
JUDGE_SYSTEM = (
    "You check the work of a web agent. You see a task and the final state of the web page. "
    "Answer Yes if the page shows that the task was accomplished, otherwise No. Answer with one "
    "word: Yes or No."
)
MAX_TOKENS = 64
MAX_INVALID_IN_A_ROW = 3


@dataclass
class Step:
    raw: str
    action: str                 # repr of the parsed action, or "invalid"
    error: str | None
    seconds: float


@dataclass
class Episode:
    task: str
    seed: int
    model: str
    budget: int
    instruction: str = ""
    steps: list[Step] = field(default_factory=list)
    stop: str = ""              # done | page_done | budget | invalid
    claim_success: bool | None = None
    claim_confidence: int | None = None
    claim_p_true: float | None = None     # P(true) at the "success" token of the report
    claim_forced: bool = False            # the report had to be asked for
    judge_p_yes: float | None = None
    judge_answer: str = ""
    page_done: bool = False
    raw_reward: float = 0.0
    reward_reason: str | None = None
    success: bool = False                 # ground truth (benchmark reward)
    n_invalid: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def user_message(obs: Observation, history: list[str]) -> str:
    past = "\n".join(history[-8:]) if history else "(none)"
    return (f"Task: {obs.instruction}\n\nPage:\n{obs.text}\n\nYour previous actions:\n{past}\n\n"
            "Next action (one JSON object):")


def _claim(gen: Generation) -> tuple[Done | None, float | None]:
    act = parse_action(gen.text)
    p_true = binary_probability(gen.tokens, {"true"}, {"false"})
    return (act if isinstance(act, Done) else None), p_true


def judge(backend: Backend, obs: Observation) -> tuple[str, float | None]:
    gen = backend.chat([{"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": f"Task: {obs.instruction}\n\nFinal page:\n"
                                                    f"{obs.text}\n\nAccomplished?"}],
                       max_tokens=2, top_logprobs=10)
    return gen.text.strip(), binary_probability(gen.tokens, {"yes"}, {"no"})


def run_episode(env: MiniWobEnv, backend: Backend, task: str, seed: int, budget: int,
                use_judge: bool = True) -> Episode:
    t0 = time.perf_counter()
    ep = Episode(task=task, seed=seed, model=backend.name, budget=budget)
    obs = env.reset(task, seed)
    ep.instruction = obs.instruction
    history: list[str] = []
    invalid_row = 0
    report: Generation | None = None
    for _ in range(budget):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message(obs, history)}]
        ts = time.perf_counter()
        gen = backend.chat(msgs, max_tokens=MAX_TOKENS)
        act = parse_action(gen.text)
        error = None
        if isinstance(act, InvalidAction):
            ep.n_invalid += 1
            invalid_row += 1
            error = f"invalid answer: {act.reason}"
            history.append(f"- (invalid) {error}")
        elif isinstance(act, Done):
            report = gen
            ep.steps.append(Step(gen.text, repr(act), None, time.perf_counter() - ts))
            ep.stop = "done"
            break
        else:
            invalid_row = 0
            error = env.execute(act, obs)
            history.append(f"- {gen.text.strip()}" + (f"  -> error: {error}" if error else ""))
        ep.steps.append(Step(gen.text, "invalid" if isinstance(act, InvalidAction) else repr(act),
                             error, time.perf_counter() - ts))
        obs = env.observe()
        if env.page_done():
            ep.stop = "page_done"
            break
        if invalid_row >= MAX_INVALID_IN_A_ROW:
            ep.stop = "invalid"
            break
    else:
        ep.stop = "budget"

    if report is None:                       # ask for the report on the final page
        ep.claim_forced = True
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message(obs, history) + "\n\n" + REPORT}]
        report = backend.chat(msgs, max_tokens=MAX_TOKENS)
    done, p_true = _claim(report)
    ep.claim_p_true = p_true
    if done is not None:
        ep.claim_success, ep.claim_confidence = done.success, done.confidence
    if use_judge:
        ep.judge_answer, ep.judge_p_yes = judge(backend, obs)
    out = env.outcome()
    ep.page_done, ep.raw_reward, ep.reward_reason = out.page_done, out.raw_reward, out.reason
    ep.success = out.success
    ep.seconds = time.perf_counter() - t0
    return ep
