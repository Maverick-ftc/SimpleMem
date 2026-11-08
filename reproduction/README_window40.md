# Primary 40-turn experiment

Status: all 48 query rows complete; independent offline validation passed.
This experiment corresponds to the final coursework report dated 8 November 2026.

## Protocol

One LoCoMo conversation (`conv-26`), all 419 dialogue turns; window 40, overlap 2;
441 memories, indexed 441/441; seed 5660 selects eight questions (two per category
1–4), repeated in three query trials. Shared fixed bank; semantic k25 versus k10.
DeepSeek `deepseek-flash`; common output cap 16384, disabled thinking, HTTP
streaming assembled as complete text, timeout 180 s, planning/reflection on,
at most two reflection rounds, sequential memory building and retrieval.

| Metric | k25 | k10 |
|---|---:|---:|
| Mean F1 (%) | 54.95 | 55.33 |
| Sample variance of trial means | 0.00003954 | 0.00046080 |
| Sample SD of trial means (pp) | 0.63 | 2.15 |
| Mean input tokens per completed question | 13,784.8 | 7,109.4 |

k10 used 48.4% fewer input tokens with a +0.38 pp difference in observed mean F1.
The first two trials favor k25; the third favors k10. This suggests a conditional
cost–variability trade-off, not a general improvement or proof of no accuracy loss.
Memory-building variance is excluded. Network retries confound latency.

## Offline demonstration

Open the project-root `verify_submission.py` in PyCharm and run it. It loads no
key and makes no API calls. It checks all saved rows and prints a compact summary.

```powershell
.venv\Scripts\python.exe verify_submission.py
.venv\Scripts\python.exe reproduction\verify_results.py --run-name locomo_window40_20261108
```

Use [the full README](README.md) for fresh installation, configuration, independent
API reruns, scoring, costs, recovery details and evidence paths. For a fresh clone,
use an unused run name rather than the recorded one; database banks are not in Git.
The current local `run_window40.py` resumes the saved experiment. Optional parsed
window caching can reuse a bank’s construction outputs and is not an independent
memory rebuild. The completed primary construction had zero cache replays.

Only the final 40-turn results are published. The original machine retains prior
pilots locally. `query_api_usage.jsonl` contains unmodified ledger lines selected
by the final rows’ API attempt IDs; the original shared budget ledger stays local.
