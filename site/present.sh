#!/bin/bash
# Serve the site for the defence and open it.
#
# The live camera needs https or localhost, so the page has to come from a
# server: opening docs/index.html by double-clicking will not get the camera.
#
#   ./site/present.sh          # port 8000
#   ./site/present.sh 9000
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8000}"

if [ ! -f docs/index.html ]; then
  echo "docs/index.html is missing — build it first:" >&2
  echo "  python3 site/build_site.py --pages --with-trials" >&2
  exit 1
fi

echo "serving docs/ at http://localhost:$PORT  (ctrl-C to stop)"
python3 -m http.server "$PORT" --directory docs >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT INT TERM
sleep 1
open "http://localhost:$PORT/"
wait $SERVER
