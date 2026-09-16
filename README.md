# Prompt Guard 86M — Adversarial Robustness Results

Word-level adversarial attacks against Meta's Prompt Guard 86M classifier (`Niansuh/Prompt-Guard-86M`), tested on `deepset/prompt-injections`. Prompt Guard correctly flags 247/263 (93.9%) injection/jailbreak prompts before attack. Each attack samples 50 of those correctly-flagged prompts and tries to rewrite them into something the model misclassifies as `BENIGN`.

## Results

| Attack | Query budget | Evaded to BENIGN | Success rate |
|---|---:|---:|---:|
| `pwws` | 1000 | 17/50 | **34.0%** |
| `textfooler` | 1000 | 31/50 | **62.0%** |
| `clare` | 1000 | 8/50 | **16.0%** |
| `pso_hownet` | 1000 | 0/50 | **0.0%** |
| `pso_wordnet` | 1000 | 6/50 | **12.0%** |
| `cea_wordnet_mlm` | 5000 | 32/50 | **64.0%** |
| `cea_textfooler` | 5000 | 39/50 | **78.0%** |

### CEA budget and regularization checks

| Attack | Query budget | Evaded to BENIGN | Success rate |
|---|---:|---:|---:|
| `cea_wordnet_mlm_q1000` | 1000 | 32/50 | **64.0%** |
| `cea_textfooler_q1000` | 1000 | 36/50 | **72.0%** |
| `cea_wordnet_mlm_reg` | 1000 | 32/50 | **64.0%** |
| `cea_textfooler_reg` | 1000 | 34/50 | **68.0%** |

Full per-example outputs (`summary.json`, `details.json`, `report.md`, `results.csv`) are in `results/<recipe>/`.

## Key findings

- CEA is the strongest attack family tested — `cea_textfooler` evades detection on 78% of prompts at 5000 queries, and still 72% at a fair 1000-query budget.
- TextFooler alone (62%) shows the model is not robust to small, targeted lexical substitutions.
- PSO is highly sensitive to candidate quality: HowNet candidates give 0%, WordNet candidates give 12%.
- Adding a modification-count regularizer to CEA trims down how much of the prompt gets rewritten, at a small cost to success rate.

## Setup

```bash
pip install -r requirements.txt
python baseline.py
python run_attack.py <recipe>
```
