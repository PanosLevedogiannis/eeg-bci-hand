#!/bin/bash
# Build και δημοσίευση του ιστότοπου στο GitHub Pages.
# Περιλαμβάνει το δημοσιευμένο υποσύνολο ΗΕΓ· βλ. site/DATA_STATEMENT.md.
set -e
cd "$(dirname "$0")/.."

python3 site/build_site.py                       # πλήρης, για το Artifact
python3 site/build_site.py --pages --with-trials  # δημόσια, για το Pages

git add docs site .gitignore
git commit -m "${1:-Update thesis presentation site}"
git push origin main

echo
echo "Έγινε push. Το Pages χτίζει σε 1-2 λεπτά:"
echo "  https://panoslevedogiannis.github.io/eeg-bci-hand/"
