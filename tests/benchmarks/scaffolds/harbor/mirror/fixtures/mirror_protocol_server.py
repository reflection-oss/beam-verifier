"""A local test server with Mirror's health and message HTTP responses."""

import http.server
import pathlib
import sys


class Handler(http.server.BaseHTTPRequestHandler):
    """Record the exact posted bytes and return the configured status."""

    def do_GET(self) -> None:  # noqa: N802 - HTTP handler method
        self.send_response(204)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler method
        pathlib.Path(sys.argv[3]).write_bytes(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(int(sys.argv[2]))
        self.end_headers()


if __name__ == "__main__":
    with http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler) as server:
        server.serve_forever()
