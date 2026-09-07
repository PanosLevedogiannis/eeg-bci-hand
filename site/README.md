# Ιστότοπος παρουσίασης διπλωματικής

- **Δημόσιος:** https://panoslevedogiannis.github.io/eeg-bci-hand/
- **Ιδιωτικός, με το πλήρες demo:** https://claude.ai/code/artifact/5f641073-7f77-4db4-a34f-1390ac3f5262

## Δύο εκδόσεις από ένα πρότυπο

| | Δημόσια (`docs/`) | Πλήρης (`site/thesis-site.html`) |
|---|---|---|
| Πρωτόκολλο, καμπύλες ERD, αποτελέσματα 14 συμμετεχόντων | ναι | ναι |
| Τρισδιάστατο χέρι με ήχο σερβοκινητήρων | ναι | ναι |
| Αναπαραγωγή δοκιμών με πραγματικό ΗΕΓ | ναι | ναι |
| Κυματομορφές | ξεχωριστό `docs/demo_data.json` | ενσωματωμένες |

Η δημόσια έκδοση ζητά το `demo_data.json` όταν φορτώνει, ώστε το υποσύνολο ΗΕΓ
να είναι ξεχωριστό, ελέγξιμο αρχείο και όχι χωμένο μέσα στη σελίδα. Αν λείπει, η
ενότητα 05 υποβαθμίζεται στο χέρι με τα χειροκίνητα κουμπιά. Τι ακριβώς περιέχει:
`DATA_STATEMENT.md`.

## Αρχεία

- `site.tpl.html` — όλη η σελίδα: HTML, CSS, γραφήματα SVG, το τρισδιάστατο
  χέρι σε three.js και η μηχανή αναπαραγωγής. Placeholders: `/*__DEMO_META__*/{}`,
  `/*__DEMO_TRIALS__*/null`, `/*__RESULTS__*/[]`, `__IMG_*__`.
- `build_site.py` — χτίζει τη μία ή την άλλη έκδοση.
- `export_demo.py` — εκπαιδεύει CSP+LDA στα runs 1-3, προβλέπει το run 4, και
  εξάγει ανά δοκιμή την αληθινή ετικέτα, την πρόβλεψη, την πιθανότητα, τις
  κυματομορφές C3/C4 στα 50 Hz και την περιβάλλουσα μ στα 25 Hz.
- `assets/` — φωτογραφίες της κατασκευής και τα συγκεντρωτικά αποτελέσματα.
  Το `assets/demo_data.json` **δεν** είναι στο git, βλ. `DATA_STATEMENT.md`.
- `publish_pages.sh` — build, commit και push της δημόσιας έκδοσης.

## Ανακατασκευή

```bash
./eeg_bci_env/bin/python site/export_demo.py site/assets/demo_data.json S11,S01,S02
python3 site/build_site.py                        # πλήρης, για το Artifact
python3 site/build_site.py --pages --with-trials  # δημόσια, για το GitHub Pages
```

Το `assets/results.json` παράγεται από τα `eeg_data/exports/reliability_summary.json`
και `classification_summary.json`.

## Μέτρηση επισκέψεων

Το GitHub Pages δεν δίνει στατιστικά. Το `Insights → Traffic` του αποθετηρίου
μετράει επισκέψεις στις σελίδες του github.com, **όχι** στον ιστότοπο.

Η σύνδεση με GoatCounter είναι έτοιμη και χρειάζεται μόνο τον κωδικό σου:

1. Λογαριασμός στο https://www.goatcounter.com — δωρεάν, διαλέγεις μια λέξη ως
   κωδικό, π.χ. `levedogiannis`.
2. Ξαναχτίζεις τη δημόσια έκδοση με τον κωδικό:
   ```bash
   GOATCOUNTER_CODE=levedogiannis python3 site/build_site.py --pages --with-trials
   git add docs && git commit -m "Enable visitor counting" && git push
   ```
3. Οι επισκέψεις φαίνονται στο `https://levedogiannis.goatcounter.com`.

Χωρίς cookies, χωρίς αποθήκευση διευθύνσεων IP, οπότε δεν χρειάζεται banner
συγκατάθεσης. Όταν είναι ενεργό, μπαίνει και σχετική σημείωση στο υποσέλιδο.
Χωρίς τη μεταβλητή δεν προστίθεται κανένα script.

## Το demo δεν είναι προσομοίωση

Κάθε δοκιμή που παίζει είναι καταγεγραμμένο σήμα και κάθε απόφαση είναι η
πραγματική έξοδος του ταξινομητή σε run εκτός εκπαίδευσης.

| Συμμετέχων | Ακρίβεια στο run 4 | LORO |
|---|---|---|
| S11 | 72,5% | Επιτυχία |
| S01 | 60,0% | Επιτυχία |
| S02 | 48,1% | Αποτυχία |
