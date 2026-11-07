# Study protocol (fixed before any model was run)

## Question

A web agent ends an episode by saying "done". Can that claim be trusted, and can cheap signals
tell a human reviewer which claims to check? A false claim of success is the costly error: the
work is reported as done and nobody looks again.

## Setup

* **Environment.** 15 MiniWoB++ tasks (Liu et al., 2018; Farama Foundation, MIT licence),
  listed with their step budgets in `src/webverify/tasks.py`, run locally in headless
  Chromium through our own Playwright harness. The task page's random generator is seeded, so a
  (task, seed) pair is one reproducible problem instance.
* **Agents.** Qwen2.5-Instruct 1.5B and 0.5B (Apache 2.0), 4-bit GGUF, on CPU with llama.cpp,
  greedy decoding. No model is fine-tuned or prompted with examples from the test problems.
* **Ground truth.** The task page's own raw reward: success iff the page ended the episode with
  a positive reward. The agent never sees it.
* **Episode.** See `src/webverify/agent.py`: at most `2 * min_steps + 2` steps; invalid answers
  are not executed; the episode ends with the agent's report (asked for if the agent did not
  give one) and an independent judge call on the final page.
* **Seeds.** Seeds 0-19 for every task and model (300 episodes per model). Seeds 0-9 are the
  CALIBRATION seeds (thresholds are chosen on them), seeds 10-19 the TEST seeds (thresholds are
  evaluated on them). If time allows, more seeds are added in blocks of 10 alternating between
  the two halves; the split rule (even block calibration, odd block test) does not change.

## Pilot (seeds 100-104, never used in the results)

Before the main run, the prompts and parsers were checked on pilot seeds 100-104 (0.5B on
100-103, 1.5B on 104). Changes made because of the pilot, all before any main-run episode:

1. The action key ``id`` was renamed ``ref``, and the HTML id attribute is shown as ``html_id``:
   the model wrote the HTML id (``"id": "options"``) instead of the element number.
2. Placeholders such as ``<ref>`` in the instructions were replaced by example values: the
   0.5B model copied ``<ref>`` literally, which made every answer invalid.
3. The final report uses its own short prompt instead of the action prompt, and its raw text is
   logged: with the action prompt the model repeated its last click instead of reporting.
4. In the report only ``success`` and ``confidence`` are read (``parse_report``): the 1.5B
   model wrote ``"action": "click"`` in 6 of 15 reports while giving both fields.

Pilot episodes are not part of any reported number.

## Research questions and analyses

| RQ | question | analysis |
|---|---|---|
| RQ1 | How often does each agent succeed? | success rate per task and overall, Wilson 95% intervals; 1.5B vs 0.5B by two-proportion z-test per task kind |
| RQ2 | When the agent says "success", how often is it wrong? | claim-vs-reward table; false-success rate (failures among success claims) with Wilson interval |
| RQ3 | Which signal best separates successes from failures, and is it calibrated? | AUROC of claim / verbal / ptrue / judge with task-level bootstrap intervals; Brier score; ECE (10 bins) |
| RQ4 | With abstention (send to a human unless claim and signal >= threshold), what risk and coverage? | threshold certified on calibration seeds for target risk r* in {5%, 10%, 20%}, delta = 5%, Bonferroni over the 20-point threshold grid; risk and coverage on test seeds |
| RQ5 | Does a bigger model help more with acting or with self-assessment? | RQ1-RQ4 for both models side by side |

Signals (definitions in `src/webverify/analysis.py`) are computed only from what the agent and
the judge produced. Nothing from the reward is used before the evaluation.

## What would change the conclusions

* Thresholds that are certified on the calibration seeds but violated on the test seeds would
  mean the exchangeability assumption between seeds fails (it should hold: same tasks, same
  generator).
* A judge that mostly answers from the task text rather than the page would show up as similar
  scores for successes and failures (AUROC near 0.5).

## Not in scope

Real websites, vision-based agents, fine-tuning, large or paid models. These are limits of the
study, stated in the README.
