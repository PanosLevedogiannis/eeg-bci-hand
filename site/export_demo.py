"""Export real replay-demo data (runs 1-3 train -> run 4 replay) to JSON for the website."""
import json, os, sys
import numpy as np
from scipy.signal import hilbert
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
import mne

PROJ = "/Users/panoslevedogiannis/Documents/eeg python protocol"
sys.path.insert(0, PROJ)
from erd_analysis import find_fif, load_and_preprocess
from classify import prepare_data, N_CSP_COMPONENTS
from reliability_analysis import get_run_ids

DATASET = "/Users/panoslevedogiannis/Downloads/inMoov_Dataset__LAST"
OUT = sys.argv[1]
SUBJECTS = sys.argv[2].split(",")
TEST_RUN = 4
DISP_T0, DISP_T1 = -1.5, 5.0
DISP_FS = 50.0
ENV_FS = 25.0

def resample_to(sig, sfreq, target_fs, smooth=None):
    """Anti-aliased decimation: boxcar smooth, then pick every step-th sample."""
    step = sfreq / target_fs
    n = int(smooth if smooth else max(1, round(step)))
    if n > 1:
        k = np.ones(n) / n
        sig = np.convolve(sig, k, mode="same")
    idx = np.round(np.arange(0, len(sig), step)).astype(int)
    idx = idx[idx < len(sig)]
    return sig[idx]

out = {}
for subj in SUBJECTS:
    subj_dir = os.path.join(DATASET, subj)
    epochs, _ = load_and_preprocess(find_fif(subj_dir), subj)
    sfreq = epochs.info["sfreq"]
    chs = epochs.ch_names
    ci = {c: chs.index(c) for c in ("C3", "C4") if c in chs}

    # display data: 1-40 Hz CAR epochs, cropped
    disp = epochs.copy().crop(tmin=DISP_T0, tmax=DISP_T1)
    disp_data = disp.get_data() * 1e6          # µV
    # mu-band envelope (8-13 Hz) for ERD meter
    mu = epochs.copy().filter(8., 13., method="fir", verbose=False).crop(tmin=DISP_T0, tmax=DISP_T1)
    mu_data = np.abs(hilbert(mu.get_data(), axis=-1)) * 1e6

    X, y, label_map = prepare_data(epochs)
    runs = get_run_ids(epochs)
    train, test = runs != TEST_RUN, runs == TEST_RUN

    clf = Pipeline([("csp", CSP(n_components=N_CSP_COMPONENTS, reg="ledoit_wolf", log=True)),
                    ("lda", LinearDiscriminantAnalysis())])
    clf.fit(X[train], y[train])
    proba = clf.predict_proba(X[test])
    preds = clf.predict(X[test])
    ytest = y[test]
    inv = {v: k for k, v in label_map.items()}
    mi_col = inv["MI"] if "MI" in inv else 0

    test_idx = np.where(test)[0]
    trials = []
    for k, i in enumerate(test_idx):
        tr = {
            "true": label_map[int(ytest[k])],
            "pred": label_map[int(preds[k])],
            "pMI": round(float(proba[k][mi_col]), 3),
        }
        for ch in ("C3", "C4"):
            if ch in ci:
                s = resample_to(disp_data[i, ci[ch]], sfreq, DISP_FS)
                tr[ch] = [int(round(v * 10)) for v in s]     # µV × 10
                e = resample_to(mu_data[i, ci[ch]], sfreq, ENV_FS, smooth=int(0.10*sfreq))
                tr[ch + "_mu"] = [int(round(v * 10)) for v in e]
        trials.append(tr)

    acc = float((preds == ytest).mean())

    # grand-average mu envelope, % ERD vs pre-cue baseline (-1.5..-0.5)
    times = disp.times
    base = (times >= -1.5) & (times <= -0.5)
    erd = {}
    for cls in ("MI", "REST"):
        sel = np.array([e[2] == epochs.event_id[cls] for e in epochs.events])
        for ch in ("C3", "C4"):
            if ch not in ci: continue
            m = mu_data[sel, ci[ch]].mean(axis=0)
            b = m[base].mean()
            pct = (m - b) / b * 100.0
            erd[f"{cls}_{ch}"] = [round(float(v), 1) for v in resample_to(pct, sfreq, ENV_FS, smooth=int(0.25*sfreq))]
    erd["t"] = [round(float(v), 3) for v in resample_to(times, sfreq, ENV_FS, smooth=1)]

    out[subj] = {
        "subject": subj,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "replay_accuracy": round(acc, 4),
        "disp_fs": DISP_FS, "env_fs": ENV_FS, "t0": DISP_T0, "t1": DISP_T1,
        "trials": trials,
        "erd": erd,
    }
    print(f"  ==> {subj}: replay acc {acc:.1%} on {int(test.sum())} unseen trials")

with open(OUT, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print("wrote", OUT, os.path.getsize(OUT) / 1e6, "MB")
