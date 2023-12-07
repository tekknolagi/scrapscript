#!/usr/bin/env python3.10
import argparse
import http.server
import logging
import socketserver
import urllib.request
import urllib.parse
from typing import Union

from scrapscript import Record, STDLIB, String, Apply, eval_exp, parse, tokenize


logger = logging.getLogger(__name__)


handler = parse(tokenize("x -> x"))


class ScrapReplServer(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        logger.debug("GET %s", self.path)
        parsed_path = urllib.parse.urlsplit(self.path)
        req = Record({"path": String(parsed_path.path)})
        res = eval_exp(STDLIB, Apply(handler, req))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(str(res).encode("utf-8"))
        return


def serve_command(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    server: Union[type[socketserver.TCPServer], type[socketserver.ForkingTCPServer]]
    if args.fork:
        server = socketserver.ForkingTCPServer
    else:
        server = socketserver.TCPServer
    server.allow_reuse_address = True
    with server(("", args.port), ScrapReplServer) as httpd:
        print("serving at port", args.port)
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    serve = subparsers.add_parser("serve")
    serve.set_defaults(func=serve_command)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--debug", action="store_true")
    serve.add_argument("--fork", action="store_true")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
