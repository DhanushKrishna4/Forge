#!/bin/sh
set -eu
og_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
og_python="$og_root/tools/og/.venv/bin/python"
if [ ! -x "$og_python" ]; then
  python3 -m venv "$og_root/tools/og/.venv"
fi
if ! "$og_python" -c 'import PIL; assert PIL.__version__ == "12.3.0"' 2>/dev/null; then
  "$og_python" -m pip install --disable-pip-version-check -r "$og_root/tools/og/requirements.txt"
fi
exec "$og_python" "$og_root/tools/og/render.py" "$@"
