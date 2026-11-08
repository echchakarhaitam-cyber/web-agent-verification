---
title: Web Agent Replay
emoji: 🕵️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Replay 600 episodes of a local web agent vs the truth
---

# Can you trust a small web agent when it says "done"? Episode replay

The 600 main-run episodes of the study (Qwen2.5 0.5B and 1.5B on 15 MiniWoB++ tasks, 20 seeds
each): for every episode, the agent's actions and raw answers, its final report, the
verification signals and the page's own reward. The models do not run here; this is a replay of
the recorded logs.

Code, protocol and analysis: https://github.com/echchakarhaitam-cyber/web-agent-verification
(this Space's files are kept in its `space/` folder; the page loads `demo/episodes.json` from a
pinned commit).
