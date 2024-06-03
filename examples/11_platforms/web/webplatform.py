#!/usr/bin/env python3.10
# Run from the root of the repository with
# $ PYTHONPATH=. python3 examples/11_platforms/web/webplatform.py
import http.server
import socketserver

from scrapscript import eval_exp, Apply, Record, parse, tokenize, String, Int

HANDLER = eval_exp(
    {},
    parse(
        tokenize(
            """
| { method = "GET", ...rest } -> get rest
| _ -> (status 501 <| page "unsupported method")

. get =
  | { path = "/" } -> (status 200 <| page "you're on the index")
  | { path = "/about" } -> (status 200 <| page "you're on the about page")
  | _ -> notfound

. notfound = (status 404 <| page "not found")
. status = code -> body -> { code = code, body = body }
. page = body -> "<!doctype html><html><body>" ++ body ++ "</body></html>"
"""
        )
    ),
)


class WebPlatform(http.server.SimpleHTTPRequestHandler):
    def handle_request(self, verb: str) -> None:
        request = Record({"method": String(verb), "path": String(self.path)})
        result = eval_exp({}, Apply(HANDLER, request))
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
        self.handle_request("GET")

    def do_POST(self) -> None:
        self.handle_request("POST")


server = socketserver.TCPServer
server.allow_reuse_address = True
with server(("", 8000), WebPlatform) as httpd:
    host, port = httpd.server_address
    print(f"serving at http://{host!s}:{port}")
    httpd.serve_forever()
