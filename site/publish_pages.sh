#!/bin/bash
# Δημοσίευση του ιστότοπου στο GitHub Pages.
#
# Χτίζει τη ΔΗΜΟΣΙΑ έκδοση, χωρίς τις κυματομορφές των συμμετεχόντων.
# Για να μπει και η ζωντανή αναπαραγωγή δοκιμών, διάβασε πρώτα το
# site/DATA_NOTICE.md και μετά τρέξε:
#   python3 site/build_site.py --pages --with-trials
#   git add -f docs/demo_data.json && git commit -m "Add demo waveforms" && git push
set -e
cd "$(dirname "$0")/.."

python3 site/build_site.py --pages

git add docs site .gitignore
git commit -m "${1:-Update thesis presentation site}"
git push origin main

echo
echo "Έγινε push."
echo "Αν είναι η πρώτη φορά, μία ρύθμιση από τον browser:"
echo "  https://github.com/PanosLevedogiannis/eeg-bci-hand/settings/pages"
echo "  Source: Deploy from a branch · Branch: main · Folder: /docs · Save"
echo
echo "Σε 1-2 λεπτά ζωντανό στο:"
echo "  https://panoslevedogiannis.github.io/eeg-bci-hand/"
