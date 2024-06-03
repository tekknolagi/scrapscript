#!/usr/bin/env python3.10
# Run from the root of the repository with
# $ PYTHONPATH=. python3 examples/11_platforms/web/webplatform.py
import http.server
import os
import socketserver

from scrapscript import eval_exp, Apply, Record, parse, tokenize, String, Int

HANDLER_FILE_NAME = "handler.scrap"
PLATFORM_DIR_NAME = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(PLATFORM_DIR_NAME, HANDLER_FILE_NAME), "r") as f:
    HANDLER = eval_exp({}, parse(tokenize(f.read())))


class WebPlatform(http.server.SimpleHTTPRequestHandler):
    def handle_request(self) -> None:
        result = eval_exp({}, Apply(HANDLER, String(self.path)))
        assert isinstance(result, Record)
        assert "code" in result.data
        assert isinstance(result.data["code"], Int)
        assert "body" in result.data
        assert isinstance(result.data["body"], String)
        self.send_response(result.data["code"].value)
        # TODO(max): Move content-type into scrapscript code
        # TODO(max): Serve scrap objects over the wire as
        # application/scrapscript
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(result.data["body"].value.encode("utf-8"))

    def do_GET(self) -> None:
        self.handle_request()


server = socketserver.TCPServer
server.allow_reuse_address = True
with server(("", 8000), WebPlatform) as httpd:
    host, port = httpd.server_address
    print(f"serving at http://{host!s}:{port}")
    httpd.serve_forever()
