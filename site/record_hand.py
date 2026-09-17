"""Drive the physical hand on the site's clock, for filming the demo clip.

    ./eeg_bci_env/bin/python site/record_hand.py                 # S11, 12 trials
    ./eeg_bci_env/bin/python site/record_hand.py --trials 10 --speed 2
    ./eeg_bci_env/bin/python site/record_hand.py --dry-run       # no Arduino

The page plays the same trials beside the clip, so the clip only needs to show
the hand. For the two to line up, the hand has to move on exactly the schedule
the page's replay engine uses:

  one trial      (T1 - T0) + HOLD seconds of trial time, divided by --speed
  command        when the trial clock reaches REVEAL, i.e. when the page
                 shows the verdict and its 3D hand moves

The decisions come from assets/demo_data.json, the classifier output that
export_demo.py stored (CSP+LDA, runs 1-3 -> run 4) and that the page shows.
Reading them from there, rather than retraining here, is what guarantees the
physical hand and the page agree trial for trial.

Before the first trial the hand makes a sync gesture: it closes at t = 0 and
opens again at t = 1 s. Trim the recording so it starts at that close, and the
clip's clock is the page's clock:

    python3 site/prepare_video.py ~/Movies/demo.mov --start <seconds to the close>

The schedule is written to assets/hand_demo.sync.json, which build_site.py
bakes into the page.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

# must match the replay engine in site.tpl.html (tick loop)
T0, T1 = -1.5, 5.0     # trial window on the page, seconds around the cue
REVEAL = 3.5           # verdict, and the command to the hand
HOLD = 1.1             # pause after the trace before the next trial

SYNC_CLOSE, SYNC_OPEN = 0.0, 1.0
PREROLL = 3.0          # first trial starts here; the hand is open again by then
LEAD_IN = 5.0          # countdown before the sync gesture, to start the camera


class Hand:
    def __init__(self, port, dry):
        self.ser = None
        if dry:
            return
        import serial
        self.ser = serial.Serial(port, 9600, timeout=2)
        time.sleep(2)                          # the board resets when the port opens
        self.ser.reset_input_buffer()
        print("  Arduino on", port)

    def send(self, label):
        if self.ser is None:
            return
        self.ser.write(b"min\n" if label == "MI" else b"mid\n")
        self.ser.readline()                    # OPEN / CLOSED, ~0.5 s

    def close(self):
        if self.ser is not None:
            self.send("REST")
            self.ser.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="S11")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--speed", type=float, default=2.0,
                    help="trial time per second, as the page's speed buttons (default 2)")
    ap.add_argument("--port", default="/dev/cu.usbmodem1101")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule, no hand")
    args = ap.parse_args()

    demo = json.load(open(os.path.join(ASSETS, "demo_data.json"), encoding="utf-8"))
    if args.subject not in demo:
        sys.exit("no %s in demo_data.json (has %s)" % (args.subject, ", ".join(demo)))
    trials = demo[args.subject]["trials"][:args.trials]
    n = len(trials)

    period = (T1 - T0 + HOLD) / args.speed
    reveal = (REVEAL - T0) / args.speed
    events = [(SYNC_CLOSE, "MI", "sync"), (SYNC_OPEN, "REST", "sync")]
    for k, tr in enumerate(trials):
        events.append((PREROLL + k * period + reveal, tr["pred"], k))
    length = PREROLL + n * period

    sync = {"subject": args.subject, "trials": n, "speed": args.speed,
            "preroll": PREROLL, "sync_close": SYNC_CLOSE}
    if not args.dry_run:                       # a rehearsal must not retime the page
        path = os.path.join(ASSETS, "hand_demo.sync.json")
        json.dump(sync, open(path, "w"), indent=2)
        print("  wrote", path)
    print("  %s, %d trials at %gx: %.1f s per trial, %.0f s in all\n"
          % (args.subject, n, args.speed, period, length))

    hand = Hand(args.port, args.dry_run)
    try:
        hand.send("REST")
        for s in range(int(LEAD_IN), 0, -1):
            print("  start recording ... %d" % s, end="\r", flush=True)
            time.sleep(1)
        print(" " * 40, end="\r")

        start = time.monotonic()               # absolute schedule: no drift
        right = 0
        for at, label, k in events:
            wait = start + at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            late = time.monotonic() - start - at
            hand.send(label)
            if k == "sync":
                print("  %6.2f s  sync  %s" % (at, "close" if label == "MI" else "open"))
                continue
            tr = trials[k]
            ok = tr["pred"] == tr["true"]
            right += ok
            print("  %6.2f s  trial %2d  cue %-4s  pred %-4s  %s%s"
                  % (at, k + 1, tr["true"], tr["pred"], "ok" if ok else "X ",
                     "   (%.2f s late)" % late if late > 0.25 else ""))
        rest = start + length - time.monotonic()
        if rest > 0:
            time.sleep(rest)
        print("\n  %d/%d correct. Stop the recording." % (right, n))
        print("  Then: python3 site/prepare_video.py <file> --start <seconds to the sync close>")
    finally:
        hand.close()


if __name__ == "__main__":
    main()
