# Models and benchmark licences

| item | source | licence | how it is used |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct, GGUF q4_k_m | huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF, file `qwen2.5-1.5b-instruct-q4_k_m.gguf` | Apache 2.0 | agent and judge, run locally with llama.cpp |
| Qwen2.5-0.5B-Instruct, GGUF q4_k_m | huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF, file `qwen2.5-0.5b-instruct-q4_k_m.gguf` | Apache 2.0 | agent and judge, run locally with llama.cpp |
| MiniWoB++ task pages | PyPI package `miniwob==1.0` (Farama Foundation) | MIT | task pages loaded from the installed package; not redistributed |

Model files are not committed (`*.gguf` and `models/` are gitignored). Put them in a directory
and pass the file path to `scripts/run_episodes.py --model`. The SHA-256 of each file used in the
committed results is recorded in the results summary.

No external model API is called anywhere in this repository.
