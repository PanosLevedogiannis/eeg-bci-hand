"""
=============================================================================
  Replay Demo — recorded EEG trials → classifier → InMoov hand
=============================================================================
  Proves the recorded dataset actually drives the hand, without needing the
  Cyton headset on someone's head.

  Honest protocol: the model is trained on runs 1-3 and replays run 4 only,
  so every trial the hand reacts to is data the classifier has never seen.

  Usage:
    python replay_demo.py --subject S11
    python replay_demo.py --subject S11 --no-arduino     # dry run, no hand
    python replay_demo.py --subject S11 --speed 0.5      # slower playback
=============================================================================
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erd_analysis import find_fif, load_and_preprocess
from classify import prepare_data, N_CSP_COMPONENTS
from reliability_analysis import get_run_ids

DATASET_DIR  = "/Users/panoslevedogiannis/Downloads/inMoov_Dataset__LAST"
ARDUINO_PORT = "/dev/cu.usbmodem1101"
ARDUINO_BAUD = 9600
HOLD_S       = 1.5      # seconds the hand holds each posture


def connect_arduino(port, baud=ARDUINO_BAUD):
    import serial
    try:
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)                      # board resets on connect
        ser.reset_input_buffer()
        print(f"  Arduino connected on {port}")
        return ser
    except Exception as e:
        print(f"  Arduino not available ({e}) — continuing without the hand")
        return None


def send_posture(ser, label):
    """MI -> close the hand, REST -> open it."""
    if ser is None:
        return
    ser.write(b"min\n" if label == "MI" else b"mid\n")
    ser.readline()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="S11")
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--port", default=ARDUINO_PORT)
    parser.add_argument("--no-arduino", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed multiplier (0.5 = half speed)")
    parser.add_argument("--test-run", type=int, default=4,
                        help="which run to hold out and replay")
    args = parser.parse_args()

    subj_dir = os.path.join(args.dataset, args.subject)
    if not os.path.isdir(subj_dir):
        print(f"No such subject folder: {subj_dir}")
        return

    epochs, _ = load_and_preprocess(find_fif(subj_dir), args.subject)
    X, y, label_map = prepare_data(epochs)
    runs = get_run_ids(epochs)

    train = runs != args.test_run
    test  = runs == args.test_run
    if test.sum() == 0:
        print(f"Run {args.test_run} has no surviving epochs for {args.subject}")
        return

    print(f"\n  Training on runs {sorted(set(runs[train]))} "
          f"({train.sum()} trials), replaying run {args.test_run} "
          f"({test.sum()} trials)")

    clf = Pipeline([
        ("csp", CSP(n_components=N_CSP_COMPONENTS, reg="ledoit_wolf", log=True)),
        ("lda", LinearDiscriminantAnalysis()),
    ])
    clf.fit(X[train], y[train])

    arduino = None if args.no_arduino else connect_arduino(args.port)

    print(f"\n  {'Trial':>5s} {'True':>6s} {'Pred':>6s} {'Hand':>7s}  {'Running acc':>12s}")
    print("  " + "-" * 46)

    correct = 0
    for i, (xi, yi) in enumerate(zip(X[test], y[test]), start=1):
        pred      = int(clf.predict(xi[np.newaxis])[0])
        true_name = label_map[int(yi)]
        pred_name = label_map[pred]
        correct  += int(pred == yi)

        send_posture(arduino, pred_name)
        mark = "ok" if pred == yi else "X"
        print(f"  {i:5d} {true_name:>6s} {pred_name:>6s} "
              f"{'CLOSE' if pred_name == 'MI' else 'OPEN':>7s}  "
              f"{correct/i:11.1%} {mark}")

        time.sleep(HOLD_S / max(args.speed, 0.01))

    n = int(test.sum())
    print("  " + "-" * 46)
    print(f"\n  Replay accuracy on unseen run {args.test_run}: "
          f"{correct}/{n} = {correct/n:.1%}")

    if arduino is not None:
        send_posture(arduino, "REST")      # leave the hand open
        arduino.close()


if __name__ == "__main__":
    main()
