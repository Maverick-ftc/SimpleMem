# SimpleMem partial reproduction for FTEC5660

The primary experiment is `locomo_window40_20261108`. It corresponds to the final
report completed on 8 November 2026. Only the final 40-turn experiment is published in this coursework fork. Earlier
local pilot results are excluded from the public submission.

## Target and scope

Run the authors’ text-memory pipeline on public LoCoMo dialogue and reproduce its
multi-step memory-based answering behavior. Then measure the effect of reducing
`semantic_top_k` from the released default 25 to 10, holding all other conditions
fixed. This is a partial reproduction rather than a match to the full paper table.

- Upstream: https://github.com/aiming-lab/SimpleMem
- Starting source commit: `db80b6a7c591e0ea730a058e9f5fc4eb06572299`.
- Paper: https://arxiv.org/abs/2601.02553v1, Table 1: GPT-4.1-mini SimpleMem
  Average F1 43.24 on the paper’s evaluation. Our baseline is 54.95 on eight fixed
  questions with DeepSeek. These values are **not directly comparable** because
  the model, task distribution, repository version and evaluation conditions differ.
- OmniSimpleMem, EvolveMem and MCP are outside the measured text experiment.

## Primary measured results

All 419 turns in conversation 0 (`conv-26`) built a single 441-entry bank with
window 40 and overlap 2. Full-text coverage was verified as 441/441. Two questions
from each category 1–4 were selected before scoring with seed 5660. Category 5 was
excluded. The eight original QA indices are 3, 56, 9, 31, 27, 30, 139 and 151.
Both arms repeated queries on the same frozen bank in three trials, yielding
8 × 2 × 3 = 48 completed query rows. There are **8 distinct questions**, not 24
independent tasks per arm. The seed controls subset/order, not provider decoding.

| Metric | Baseline k25 | Modified k10 |
|---|---:|---:|
| Mean F1 (%) | 54.95 | 55.33 |
| Sample variance of trial means (0–1 scale) | 0.00003954 | 0.00046080 |
| Sample SD of trial means (percentage points) | 0.63 | 2.15 |
| Mean input tokens per question | 13,784.8 | 7,109.4 |
| Mean output tokens per question | 780.3 | 739.8 |
| Mean retrieved entries | 54.92 | 26.71 |
| Mean seconds per question, affected by retries | 36.64 | 31.42 |
| Completed rows | 24/24 | 24/24 |

Trial mean F1 (%): k25 = **55.68, 54.65, 54.53**; k10 = **55.02, 53.36, 57.62**.
The observed overall difference is +0.38 F1 percentage points, with 48.4% fewer
input tokens. The first two trials favor k25 and the third favors k10. The defensible
finding is similar observed mean lexical F1 at lower input usage on this subset,
with greater observed trial variability. No significance, equivalence,
non-inferiority or general accuracy improvement is established.

Input/output counts include all successful planning, reflection and answer calls
attributed to each completed question; the one-time memory build is excluded.
Semantic k is a per-search parameter, not a cap on the final merged entry count.
Timing is confounded by network retries and process recovery. Trial SD excludes
memory-building variance because the bank was constructed once.

## Environment and common conditions

Windows/PyCharm; Python 3.12.4; CPU embeddings; hosted DeepSeek API; no local GPU.
Pinned text dependencies are in `requirements-text.txt`; the observed environment
is recorded in `requirements-lock.txt` and each saved `plan.json`.

- Model/provider: `deepseek-flash`, DeepSeek; base URL `https://api.deepseek.com`.
  This is a provider alias, not an immutable model checkpoint.
- Temperature: memory calls 0.1; query stages use 0.1 / 0.2 / 0.3 as in the original
  code. `top_p` is not explicitly sent; no API decoding seed; thinking disabled.
- Output cap 16,384; HTTP streaming assembled into a complete agent response;
  180-second timeout; SDK retries disabled. Upstream retries are separately metered.
- Planning and reflection enabled; at most two reflection rounds; both forms of
  parallel processing disabled. Author prompt templates and core files retained.
- Local `Qwen/Qwen3-Embedding-0.6B`, normalized standard SentenceTransformer
  embeddings, 1,024 dimensions. Revision:
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.

## Fresh installation on Windows

Run these commands from the repository root in a PowerShell terminal. In PyCharm,
open the repository itself as the project; select `.venv\Scripts\python.exe` as
the interpreter and the repository root as working directory.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r reproduction\requirements-text.txt
.venv\Scripts\python.exe reproduction\download_data.py
.venv\Scripts\python.exe reproduction\download_embedding.py
```

The data downloader verifies the official commit and SHA-256; the model downloader
verifies publisher hashes for approximately 1.2 GB of weights. Neither calls the
language-model API. A complete dependency installation is needed for a fresh API
experiment; the saved-results validator itself uses Python’s standard library.

**For API experiments only**, create the local configuration on a fresh clone:

```powershell
Copy-Item reproduction\config_deepseek.py.example config.py
Copy-Item reproduction\deepseek.env.example .env
```

Then enter your key locally under `DEEPSEEK_API_KEY` in `.env`. The template defaults
to the official DeepSeek endpoint and `deepseek-flash`. Do not overwrite an existing
local configuration. Do not commit, display or record `.env` or root `config.py`.

## Offline check and presentation demonstration

After downloading the official data, run **`verify_submission.py`** directly in
PyCharm (right-click its editor tab and select Run). It does not load the key,
construct models or call an API. It validates the saved evidence and prints the
eight questions, trial scores, means, SDs and input-token reduction.

Equivalent terminal commands:

```powershell
.venv\Scripts\python.exe verify_submission.py
.venv\Scripts\python.exe reproduction\verify_results.py --run-name locomo_window40_20261108
.venv\Scripts\python.exe reproduction\test_offline.py
.venv\Scripts\python.exe -m reproduction.test_rerun_safety
```

The validator checks official references, paired-row uniqueness, F1/statistics,
original core hashes, frozen-memory checksum, index coverage, retrieved-memory IDs,
normal API termination and per-question usage reconciliation. The saved primary
validation passed for all 48 rows and 314 successful query responses, including
13 recovered failed attempts attributed to completed rows. Validation makes zero
API calls and does not claim that all answers are factually correct.

## Execute a new independent API experiment

The root `run_window40.py` records the original local entry settings and resumes
the named run on the original machine. **Do not use its recorded run name on a
clean clone**: local database files are intentionally excluded from Git, whereas
the completion markers are retained as evidence. Use an unused name instead.

```powershell
.venv\Scripts\python.exe reproduction\run_experiment.py --mode experiment --execute --samples 0 --per-category 2 --trials 3 --seed 5660 --run-name window40_fresh_01 --memory-window-size 40 --max-output 16384 --api-timeout 180 --budget-cny 15 --console-log
```

Choose another unused name if `window40_fresh_01` already exists. This command omits
`--cache-memory-windows` to construct an independent bank. Adding that flag allows
reuse of matching parsed windows and therefore does not guarantee an independent
memory rebuild. The completed primary bank recorded **zero** cache replays.

The budget guard is cumulative over the local shared `results/api_usage.jsonl`.
This full historical ledger is kept locally and ignored by Git. On a fresh clone
the ledger starts empty; the CNY 15 setting is a maximum guard, not a required
expense. On the original machine the archived guard total was CNY 10.92569, which
includes earlier attempts and unknown failed-request reservations. Successful
usage with known tokens across that local history was about CNY 4.804 at the
conservative peak rates; neither number is a provider invoice. Do not clear an
existing ledger to present misleading accounting. The implementation rejects
guard caps above the original CNY 30 authorization.

The public run directory contains `query_api_usage.jsonl`: unchanged original
ledger lines selected by the final 48 rows’ `api_attempt_ids`, retaining the
original IDs, successful responses, errors, reservations and adjustments. It
contains only this primary comparison’s query evidence, not earlier pilot calls
or the one-time memory-build calls. The validator uses this per-run file when
available; a new local run can instead use the local shared ledger.

Existing complete questions are skipped on resumption. A partial bank without a
completion marker requires a new run name. Empty, truncated or missing-usage
responses stop rather than silently becoming valid results. Three consecutive
API errors stop the run. No additional API run is necessary for the presentation.

## Modification and code map

- `main.py`, `simplemem/core/memory_builder.py`, `hybrid_retriever.py` and
  `answer_generator.py`: original author pipeline; core hashes recorded in plans.
- `reproduction/run_experiment.py`: paired protocol. Search `EXPERIMENT_VARIANTS`
  for the k25/k10 treatment and `system.hybrid_retriever.semantic_top_k = k` for
  its application. `refresh_full_text_index` is a common correctness adaptation.
- `reproduction/common.py`: deterministic subset and pinned set-token F1.
- `reproduction/api_meter.py`: metering, budgets and HTTP response assembly.
- `reproduction/memory_window_cache.py`: parsed-memory recovery, not answer reuse.
- `verify_submission.py`: readable offline demonstration.
- `reproduction/verify_results.py`: independent evidence checks.

The scorer lowercases text, replaces only `. , ! ?` with spaces and deduplicates
whitespace-separated tokens. Precision/recall are computed from the token-set
intersection; F1 is their harmonic mean, zero for empty/disjoint sets. No article
removal or stemming is applied. It matches the pinned SimpleMem evaluator and is
not automatically identical to the original LoCoMo evaluation. Each trial averages
eight question scores; sample variance of the three trial means uses denominator 2.

## Debug diary and limitations

Early requests suffered connection/protocol failures and an initial 8,192-token
truncation. The final common transport/output settings and resumption safeguards
enabled completion with the released 40-turn window. Logs do not establish whether
network, proxy, provider or request length was the underlying cause.

The preliminary 10-turn bank had 272 indexed documents against 582 memories.
Fifteen pre-fix query rows were archived locally and excluded; they are not part
of this public submission. Both final arms rebuild the index after construction. The primary
40-turn bank has complete 441/441 coverage. Only the final 40-turn evidence is published here.

Eight questions from one conversation cannot support full-benchmark rankings or
prove preserved accuracy. There is no no-memory/full-context baseline, no test of
the paper’s claimed 30-fold savings, and no repeated independent memory building.
Provider aliases and unseeded generation prevent guaranteed identical reruns.

## Evidence and submission links

Primary directory: `results/locomo_window40_20261108/`:
`plan.json`, `answers.jsonl`, `summary.json`, `memory_0.json`,
`memory_0_entries.json`, `index_0.json`, `validation.json`, `api_response_map.json`,
`query_api_usage.jsonl`. The public per-run ledger contains usage and error
accounting, not keys, headers or prompts. Data/model provenance is in `data/`. Local `.env`, root
`config.py`, database banks, model caches, logs and IDE files are ignored.

Repository: https://github.com/Maverick-ftc/SimpleMem

The final PDF and the recorded Zoom presentation are separate course submissions.
The signed report is not published in this public code repository. The recording
must be at most 15 minutes; its link is supplied after the student records it.
