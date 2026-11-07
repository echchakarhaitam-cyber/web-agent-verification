# Background: agents, self-reports and when to trust them

## The setting

A web agent receives an instruction ("Enter the username ... and press login"), sees the page,
and acts step by step. At the end it reports: done, successfully or not. In practice that report
decides whether a person looks at the work again. Two errors are possible:

* a **false success**: the agent says it succeeded, it did not; nobody checks, the error stays;
* a **missed success**: the agent says it failed although it succeeded; a person checks for
  nothing.

The first is the costly one, so the study measures, among episodes reported as successful, the
share that actually failed (the *false-success rate*), against the benchmark's own reward.

## MiniWoB++

MiniWoB++ (Shi et al., 2017; Liu et al., 2018; maintained by the Farama Foundation) is a set of
small synthetic web pages, each with a natural-language instruction and a programmatic reward:
the page itself checks whether the task was done. Each page is generated from a random seed, so
a (task, seed) pair is a reproducible problem. Because the reward is computed by the page, it is
an independent ground truth that the agent cannot influence by what it says.

## Signals that an episode succeeded

Four numbers are computed per episode, none of which uses the reward:

* the **claim** (yes/no) and the **verbal confidence** (0-100) that the agent writes in its
  report. Language models are known to state high confidence regardless of correctness, and
  verbal confidence is often poorly calibrated (Xiong et al., 2024);
* the **token probability** of `true` in the report. This is the model's own probability
  distribution at the moment it writes its verdict; it can carry information the verbal number
  does not (Kadavath et al., 2022);
* a **judge**: the same model, asked in a fresh prompt without the agent's history or claim
  whether the final page shows the task done. Using a model to evaluate outputs ("LLM as a
  judge") is common (Zheng et al., 2023); here it is a weak judge (the same small model), which
  is the realistic setting for a local agent.

**AUROC** measures how well a signal ranks successes above failures (0.5 = no information,
1 = perfect); the **Brier score** and **ECE** measure whether its values are calibrated
probabilities.

## Abstention with a certified risk

Rule: accept an episode automatically only if the agent claims success **and** the signal is at
least a threshold t; otherwise send it to a person. The *coverage* is the share of episodes
accepted, the *risk* the share of accepted episodes that failed. Raising t lowers the risk and
the coverage.

To pick t without overfitting, it is chosen on the calibration seeds and evaluated on the test
seeds. For each candidate t of a grid fixed in advance (0.00, 0.05, ..., 0.95), with n accepted
calibration episodes of which k failed, the hypothesis "risk > r*" is rejected when the
binomial probability P(Bin(n, r*) <= k) is at most delta / 20. The loosest rejected threshold is
kept. By the union bound, the probability of keeping a threshold whose true risk exceeds r* is
at most delta, if calibration and test episodes are exchangeable (same tasks, different seeds).
This is the multiple-testing approach to risk control of Angelopoulos et al. (2021), applied to
selective prediction (Geifman & El-Yaniv, 2017).

## References

* Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L. (2021). Learn then
  test: calibrating predictive algorithms to achieve risk control. arXiv:2110.01052.
* Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks.
  NeurIPS.
* Kadavath, S., et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221.
* Liu, E. Z., Guu, K., Pasupat, P., Shi, T., & Liang, P. (2018). Reinforcement learning on web
  interfaces using workflow-guided exploration. ICLR.
* Shi, T., Karpathy, A., Fan, L., Hernandez, J., & Liang, P. (2017). World of Bits: an
  open-domain platform for web-based agents. ICML.
* Xiong, M., et al. (2024). Can LLMs express their uncertainty? An empirical evaluation of
  confidence elicitation in LLMs. ICLR.
* Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS.
* Qwen Team (2024). Qwen2.5 technical report. arXiv:2412.15115.
