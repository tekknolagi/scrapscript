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
    def do_GET(self) -> None:
        logger.debug("GET %s", self.path)
        parsed_path = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed_path.query)
        logger.debug("PATH %s", parsed_path)
        logger.debug("QUERY %s", query)
        if parsed_path.path == "/repl":
            return self.do_repl()
        if parsed_path.path == "/scrapscript.py":
            return self.do_scrapscript_py()
        if parsed_path.path == "/compiler.py":
            return self.do_compiler_py()
        if parsed_path.path == "/eval":
            try:
                return self.do_eval(query)
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
                return
        if parsed_path.path == "/style.css":
            self.send_response(200)
            self.send_header("Content-type", "text/css")
            self.end_headers()
            with open(os.path.join(ASSET_DIR, "style.css"), "rb") as f:
                self.wfile.write(f.read())
            return
        return self.do_404()

    def do_repl(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.end_headers()
        with open(os.path.join(ASSET_DIR, "repl.html"), "rb") as f:
            self.wfile.write(f.read())
        return

    def do_scrapscript_py(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        with open(os.path.join(ASSET_DIR, "scrapscript.py"), "rb") as f:
            self.wfile.write(f.read())
        return

    def do_compiler_py(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        with open(os.path.join(ASSET_DIR, "compiler.py"), "rb") as f:
            self.wfile.write(f.read())
        return

    def do_404(self) -> None:
        self.send_response(404)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"""try hitting <a href="/repl">/repl</a>""")
        return

    def do_eval(self, query: Dict[str, Any]) -> None:
        exp = query.get("exp")
        if exp is None:
            raise TypeError("Need expression to evaluate")
        if len(exp) != 1:
            raise TypeError("Need exactly one expression to evaluate")
        exp = exp[0]
        tokens = tokenize(exp)
        ast = parse(tokens)
        env = query.get("env")
        if env is None:
            env = STDLIB
        else:
            if len(env) != 1:
                raise TypeError("Need exactly one env")
            env_object = deserialize(env[0])
            assert isinstance(env_object, EnvObject)
            env = env_object.env
        logger.debug("env is %s", env)
        monad = ScrapMonad(env)
        result, next_monad = monad.bind(ast)
        serialized = EnvObject(next_monad.env).serialize()
        encoded = bencode(serialized)
        response = {"env": encoded.decode("utf-8"), "result": str(result)}
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))
        return


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
