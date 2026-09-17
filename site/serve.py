"""Static file server for the presentation, with HTTP byte ranges.

    python3 site/serve.py [PORT] [DIRECTORY]      # defaults: 8000, docs/

python -m http.server ignores Range headers. Chrome copes, but Safari will not
play a <video> from a server that does not answer them with 206, so the demo
clip would stay black on the presenting Mac. This adds exactly that.
"""
import http.server
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


class Handler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        m = RANGE.match(self.headers.get("Range", "").strip())
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()

        size = os.path.getsize(path)
        first, last = m.groups()
        if first:
            start, end = int(first), int(last) if last else size - 1
        elif last:                                   # "bytes=-N": the last N bytes
            start, end = max(0, size - int(last)), size - 1
        else:
            return super().send_head()
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", "bytes */%d" % size)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        self._left = None
        while left > 0:
            chunk = source.read(min(64 * 1024, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)

    def end_headers(self):
        if self.command in ("GET", "HEAD") and not self._headers_buffer_has("Accept-Ranges"):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def _headers_buffer_has(self, name):
        key = name.lower().encode("latin-1")
        return any(line.lower().startswith(key) for line in getattr(self, "_headers_buffer", []))

    def log_message(self, *args):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    directory = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "docs")
    handler = lambda *a, **k: Handler(*a, directory=directory, **k)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print("serving %s at http://localhost:%d" % (directory, port))
        httpd.serve_forever()


if __name__ == "__main__":
    main()
