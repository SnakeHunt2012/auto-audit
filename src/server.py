#!/usr/bin/env python
# -*- coding: utf-8 -*-

from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler

class HTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        content = "{\"a\": 1, \"b\": 2}\n"
        self.send_response(200)
        self.send_header("Content-type", "text/html;charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

def main():

    server_address = ("", 4321)
    http_deamon = HTTPServer(server_address, HTTPRequestHandler)
    http_deamon.serve_forever()

if __name__ == "__main__":

    main()
