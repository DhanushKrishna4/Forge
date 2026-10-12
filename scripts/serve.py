"""Serve the production site at a real GitHub Pages-style repository subpath."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=4173);p.add_argument('--base',default='/Forge/')
args=p.parse_args();base='/'+args.base.strip('/')+'/'
class Handler(SimpleHTTPRequestHandler):
    extensions_map={**SimpleHTTPRequestHandler.extensions_map,'.wasm':'application/wasm','.mjs':'text/javascript'}
    def do_GET(self):
        path=urlsplit(self.path).path
        if path==base[:-1]:self.send_response(301);self.send_header('Location',base);self.end_headers();return
        if not path.startswith(base):self.send_error(404,'Use '+base);return
        self.path=self.path[len(base)-1:]
        super().do_GET()
    def end_headers(self):
        # Deliberately no COOP/COEP: Pages must work without SharedArrayBuffer.
        self.send_header('Cache-Control','no-cache')
        super().end_headers()

if not (ROOT/'web/dist/index.html').exists():p.error('Build web/dist before serving')
server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=ROOT/'web/dist'))
print(f'Forge: http://127.0.0.1:{args.port}{base}',flush=True)
server.serve_forever()
