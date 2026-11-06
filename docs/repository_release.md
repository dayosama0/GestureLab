# GitHub source release

Repository: https://github.com/dayosama0/GestureLab (private).

The initial source import is organized by component: environment, baseline ML,
expanded-data fine-tuning, prediction API, browser interface, measured artifacts,
supporting tools, and documentation. Commit dates record the actual import time.
They do not reconstruct the development chronology.

## Included and local files

The repository contains source code, tests, package locks, measured numeric
results, training curves, confusion matrices, and run instructions.

The dataset, source assignment files, trained checkpoints, uploaded photographs,
image-bearing report exports, recordings, temporary renders, and local tool
state remain outside Git. The local DOCX and PDF remain available in `report/`.
Cloning this repository alone does not install the trained model weights.
Prediction requires the matching local checkpoints or an explicit training run
with the private source dataset. See README.md and ml/README.md.

## Verification before the initial import

Executed on 7 November 2026 from the existing working project:

| Command | Actual result |
| --- | --- |
| `.venv/Scripts/python.exe -m pytest backend/tests ml/tests -q` | 50 passed in 13.66s |
| `npm --prefix frontend run typecheck` | Exit 0 |
| `npm --prefix frontend run lint` | Exit 0 |
| `npm --prefix frontend test` | 22 passed, 2 test files |
| `npm --prefix frontend run build` | Exit 0, Vite build completed in 5.08s |
| `npm --prefix frontend run test:e2e` | 10 passed in 10.2s |
| `.venv/Scripts/python.exe scripts/smoke_expansion.py` | PASS: 5 trained models, 10 actual held-out predictions, 6 measured runs, exact 224/256 input and Grad-CAM alignment; 13 legacy files unchanged |

The publication audit found no credential-pattern matches or private media,
checkpoint, or dataset archives among the selected files. Tests above used the
existing local checkpoints and dataset; they do not imply that those private
assets are present in a fresh clone. Personal iPhone evaluation, laptop transfer,
and an updated demo recording remain pending.
