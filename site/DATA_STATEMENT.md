# Δήλωση δεδομένων

Ο ιστότοπος δημοσιεύει ένα υποσύνολο ΗΕΓ, ώστε η επίδειξη να τρέχει σε πραγματικό
σήμα και όχι σε προσομοίωση. Αυτή η σελίδα λέει ακριβώς τι είναι αυτό.

## Τι δημοσιεύεται

Αρχείο: `docs/demo_data.json`

| | |
|---|---|
| Συμμετέχοντες | 3 από τους 14, με κωδικούς S01, S02, S11 |
| Δοκιμές | 224, μόνο από το run 4 κάθε συνεδρίας |
| Κανάλια | 2 από τα 8 — C3 και C4 |
| Παράθυρο | 6,5 s ανά δοκιμή, από −1,5 s ως +5,0 s γύρω από τον marker |
| Επεξεργασία | ζωνοπερατό 1–40 Hz, notch 50 Hz, κοινή μέση αναφορά |
| Ανάλυση | υποδειγματοληψία στα 50 Hz από τα 250 Hz της καταγραφής |
| Συνοδευτικά | ανά δοκιμή: η ετικέτα, η πρόβλεψη του ταξινομητή, η πιθανότητα |

Χωρίς αναγνωριστικά. Οι κωδικοί S01, S02, S11 είναι αύξοντες αριθμοί συνεδρίας
και δεν αντιστοιχίζονται σε πρόσωπα μέσα στο αποθετήριο.

## Τι δεν δημοσιεύεται

Οι ακατέργαστες καταγραφές, τα υπόλοιπα έξι κανάλια, η καταγραφή βάσης με
ανοιχτά και κλειστά μάτια, τα runs 1–3, οι έντεκα άλλοι συμμετέχοντες, και
οποιοδήποτε δημογραφικό στοιχείο ή χρονική σήμανση συνεδρίας.

## Πώς προέκυψε

Το `site/export_demo.py` εκπαιδεύει CSP με τέσσερις συνιστώσες και κανονικοποίηση
Ledoit-Wolf, μαζί με γραμμικό διαχωριστή Fisher, στα runs 1–3 κάθε συμμετέχοντα.
Έπειτα προβλέπει το run 4, που ο ταξινομητής δεν έχει δει. Εξάγονται μόνο οι
δοκιμές αυτού του run.

## Άδεια

Ο κώδικας διατίθεται με τους όρους του αποθετηρίου. Τα δεδομένα διατίθενται για
εκπαιδευτική και ερευνητική χρήση, με αναφορά στην εργασία.

---

## Data statement (English)

The site publishes a small EEG subset so the demo runs on recorded signal rather
than a simulation. It contains 224 trials from three participants, two of eight
channels (C3 and C4), 6.5 s per trial, band-passed 1–40 Hz with a common average
reference and downsampled to 50 Hz. Only run 4 of each session is included, which
is the run held out from classifier training. No identifiers, no demographics, no
raw recordings, no baseline segments. Participant codes are session numbers and
are not linked to individuals anywhere in this repository.
