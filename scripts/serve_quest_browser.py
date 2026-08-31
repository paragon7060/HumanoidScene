#!/usr/bin/env python3
"""Serve the built CloudXR sample on a trusted LAN without a webpack dev server."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import ssl


class StaticFiles(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        root = Path(self.directory).resolve()
        target = Path(super().translate_path(path)).resolve()
        if not target.is_relative_to(root):
            return str(root / "__blocked_path__")
        return str(target)

    def list_directory(self, path):
        self.send_error(403, "Directory listings are disabled")

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--key", type=Path)
    args = parser.parse_args()
    if bool(args.certificate) != bool(args.key):
        parser.error("--certificate and --key must be supplied together")
    if not (args.directory / "index.html").is_file():
        parser.error("Built index.html is missing; run npm build first")
    handler = partial(StaticFiles, directory=str(args.directory.resolve()))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        scheme = "http"
        if args.certificate:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(args.certificate, args.key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
            scheme = "https"
        print(f"[READY] Quest browser: {scheme}://{args.host}:{server.server_port}", flush=True)
        print("Trusted LAN use only; Ctrl+C stops the server.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
