# legacy/

Scripts from earlier stages of the thesis. None of them produced the results
reported in the thesis or on the site, and they are not maintained.

| Script | What it was |
|---|---|
| `load_data.py`, `preprocess.py`, `visualize.py`, `run_all.py` | First single-subject pipeline, built on the PhysioNet motor-imagery dataset. `preprocess.py` is the file quoted in Listings 4.3–4.6 of the thesis; the reported results use `erd_analysis.load_and_preprocess` instead (no ICA, 200 µV rejection). |
| `multi_subject_analysis.py` | Batch run of that pipeline over Cyton recordings. |
| `full_analysis.py` | Analysis of the first 4 participants, with its own copy of the preprocessing. |
| `run_progression.py` | Run-by-run accuracy; replaced by the run trend in `reliability_analysis.py`. |
| `statistics_analysis.py` | Mann-Whitney band-power tests and a 10-fold permutation test; replaced by the leave-one-run-out permutation test in `reliability_analysis.py`. |
| `load_recording.py` | Prints a summary of one recorded FIF file. |

They import each other and `classify.py` from the repository root, so run them
from the root with:

```bash
PYTHONPATH=. python legacy/run_all.py
```
