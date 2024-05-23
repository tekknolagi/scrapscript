#!/usr/bin/env python3.10
import argparse
import logging
import http.server
import urllib.parse
import socketserver
import os
import json
from scrapscript import ScrapMonad, parse, tokenize, deserialize, EnvObject, bencode, STDLIB
from typing import Any, Dict, Union


ASSET_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


class ScrapReplServer(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        # Enable Cross-Origin Resource Sharing (CORS)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        if code == 404:
            self.error_message_format = """try hitting <a href="/repl.html">/repl.html</a>"""
        return super().send_error(code, message)


def serve(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    server: Union[type[socketserver.TCPServer], type[socketserver.ForkingTCPServer]]
    if args.fork:
        server = socketserver.ForkingTCPServer
    else:
        server = socketserver.TCPServer
    server.allow_reuse_address = True
    with server(("", args.port), ScrapReplServer) as httpd:
        host, port = httpd.server_address
        print(f"serving at http://{host!s}:{port}")
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(prog="webrepl")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fork", action="store_true")
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
