"""Bundle the exact Python sources and optionally fetch the pinned browser runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.27.7'
BASE = f'https://cdn.jsdelivr.net/pyodide/v{VERSION}/full/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', action='store_true', help='Download Pyodide and NumPy once (~17 MB)')
    args = parser.parse_args()
    assets = ROOT/'web/public/assets'
    assets.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(assets/'forge.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT/'src/forge_ml').glob('*.py')):
            info = zipfile.ZipInfo('forge_ml/'+path.name, (2026,10,2,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info,path.read_bytes())
    if args.runtime:
        target = ROOT/'web/public/pyodide'
        target.mkdir(exist_ok=True)
        def get(name):
            path = target/name
            if not path.exists():
                print('Downloading',name,flush=True)
                with urllib.request.urlopen(BASE+name,timeout=90) as response:
                    data=response.read()
                path.write_bytes(data)
            return path
        lock=json.loads(get('pyodide-lock.json').read_text())
        names=['pyodide.mjs','pyodide.asm.js','pyodide.asm.wasm','python_stdlib.zip',lock['packages']['numpy']['file_name']]
        for name in names: get(name)
        numpy=lock['packages']['numpy']
        assert hashlib.sha256((target/numpy['file_name']).read_bytes()).hexdigest() == numpy['sha256']
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.iterdir()) if p.is_file()}
        (assets/'runtime-manifest.json').write_text(json.dumps({'pyodide':VERSION,'numpy':numpy['version'],'files':manifest},indent=2))
    print('Bundled',assets/'forge.zip')

if __name__=='__main__': main()
