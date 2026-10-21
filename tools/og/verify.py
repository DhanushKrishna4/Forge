"""Check Forge's static sharing metadata and PNG, optionally over HTTPS.

Uses only the Python standard library. No JavaScript execution is involved.
"""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import struct
import sys
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zlib

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = "https://dhanushkrishna4.github.io/Forge/"
IMAGE_URL = CANONICAL + "og-v1.png"
MAX_BYTES = 1_000_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_png(data):
    require(len(data) < MAX_BYTES, "PNG must be smaller than 1 MB")
    require(data.startswith(b"\x89PNG\r\n\x1a\n"), "Not PNG image bytes")
    chunks, offset = [], 8
    while offset < len(data):
        require(offset + 12 <= len(data), "Truncated PNG chunk")
        size = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + size
        require(end <= len(data), "Truncated PNG payload")
        payload = data[offset + 8:end - 4]
        crc = struct.unpack(">I", data[end - 4:end])[0]
        require(zlib.crc32(kind + payload) & 0xFFFFFFFF == crc,
                f"Invalid PNG CRC: {kind!r}")
        chunks.append((kind, payload))
        offset = end
        if kind == b"IEND":
            break
    names = [kind for kind, _ in chunks]
    require(names and names[0] == b"IHDR" and names[-1] == b"IEND",
            "PNG must begin with IHDR and end with IEND")
    require(offset == len(data), "Unexpected bytes after PNG end")
    require(names.count(b"IHDR") == 1 and len(chunks[0][1]) == 13,
            "Invalid PNG header")
    require(not chunks[-1][1], "Invalid PNG end chunk")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", chunks[0][1])
    require((width, height) == (1200, 630), "PNG must be exactly 1200 x 630")
    require(depth == 8 and color == 2, "PNG must be 8-bit truecolor RGB (type 2)")
    require(b"tRNS" not in names, "PNG must be opaque, without transparency")
    require(b"acTL" not in names, "PNG must be static, not animated")
    require((compression, filtering, interlace) == (0, 0, 0),
            "Expected non-interlaced PNG with standard compression/filtering")
    require(names.count(b"sRGB") == 1, "PNG must contain one explicit sRGB chunk")
    srgb = next(payload for kind, payload in chunks if kind == b"sRGB")
    require(len(srgb) == 1 and srgb[0] <= 3, "Invalid sRGB rendering intent")
    require(b"iCCP" not in names, "Unexpected competing ICC color profile")
    require(b"IDAT" in names, "PNG has no pixel data")
    require(names.index(b"sRGB") < names.index(b"IDAT"),
            "sRGB declaration must precede image data")
    compressed = b"".join(payload for kind, payload in chunks if kind == b"IDAT")
    stride, expected = width * 3 + 1, (width * 3 + 1) * height
    decoder = zlib.decompressobj()
    pixels = decoder.decompress(compressed, expected + 1)
    require(len(pixels) == expected and decoder.eof and not decoder.unused_data,
            "Invalid or incorrectly sized PNG pixel data")
    require(all(pixels[row * stride] <= 4 for row in range(height)),
            "Invalid PNG scanline filter")
    return {"dimensions": [width, height], "mode": "RGB", "bitDepth": depth,
            "opaque": True, "colorSpace": "sRGB", "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


class StaticHead(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_head = False
        self.in_title = False
        self.closed_head = False
        self.meta = {}
        self.canonical = []
        self.title = []

    def handle_starttag(self, tag, attrs):
        if tag == "head":
            require(not self.in_head and not self.closed_head, "Duplicate HTML head")
            self.in_head = True
        if not self.in_head:
            return
        values = dict(attrs)
        require(len(values) == len(attrs), f"Duplicate attribute on {tag}")
        if tag == "title":
            self.in_title = True
        elif tag == "link" and values.get("rel") == "canonical":
            self.canonical.append(values.get("href"))
        elif tag == "meta":
            key = values.get("property", values.get("name"))
            if key:
                require(key not in self.meta, f"Duplicate metadata: {key}")
                self.meta[key] = (values.get("content"),
                                  "property" if "property" in values else "name")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag == "head":
            self.closed_head = True
            self.in_head = False

    def handle_data(self, data):
        if self.in_head and self.in_title:
            self.title.append(data)


def check_html(data):
    parser = StaticHead()
    parser.feed(data.decode("utf-8"))
    parser.close()
    require(parser.closed_head, "Missing closed static HTML head")
    require(parser.canonical == [CANONICAL], "Canonical URL is absent or incorrect")
    fixed = {
        "og:type": "website", "og:site_name": "Forge", "og:url": CANONICAL,
        "og:image": IMAGE_URL, "og:image:type": "image/png",
        "og:image:width": "1200", "og:image:height": "630",
        "twitter:card": "summary_large_image", "twitter:image": IMAGE_URL,
    }
    required = list(fixed) + ["og:title", "og:description", "og:image:alt",
                              "twitter:title", "twitter:description", "twitter:image:alt"]
    for key in required:
        require(key in parser.meta and bool(parser.meta[key][0]), f"Missing {key}")
        expected_attr = "property" if key.startswith("og:") else "name"
        require(parser.meta[key][1] == expected_attr,
                f"{key} must use the {expected_attr} attribute")
    meta = {key: value[0] for key, value in parser.meta.items()}
    for key, value in fixed.items():
        require(meta[key] == value, f"Incorrect {key}")
    for suffix in ["title", "description", "image", "image:alt"]:
        require(meta["og:" + suffix] == meta["twitter:" + suffix],
                f"OG and Twitter {suffix} disagree")
    require(meta["og:title"] == "".join(parser.title).strip(),
            "Sharing title must match the document title")
    require(meta["og:description"] == meta.get("description"),
            "Sharing description must match the document description")
    for url in [CANONICAL, meta["og:image"]]:
        parts = urlsplit(url)
        require(parts.scheme == "https" and bool(parts.netloc),
                "Sharing URLs must be absolute HTTPS URLs")
    return {key: meta[key] for key in required}


def fetch(url, expected_type):
    request = Request(url, headers={"User-Agent": "Forge-OG-verifier/1.0"})
    with urlopen(request, timeout=30) as response:
        require(response.status == 200, f"Unexpected HTTP status for {url}")
        mime = response.headers.get_content_type()
        require(mime == expected_type, f"Expected {expected_type}, received {mime}: {url}")
        data = response.read(MAX_BYTES + 1)
        require(len(data) <= MAX_BYTES, f"Response exceeds 1 MB: {url}")
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=ROOT / "web/index.html")
    parser.add_argument("--image", type=Path, default=ROOT / "web/public/og-v1.png")
    parser.add_argument("--url", help="Optional published HTTPS page URL to verify")
    args = parser.parse_args()
    try:
        local_image = args.image.read_bytes()
        report = {"image": check_png(local_image), "metadata": check_html(args.html.read_bytes()),
                  "local": "passed", "live": "not requested"}
        if args.url:
            require(args.url == CANONICAL, f"Hosted URL must be {CANONICAL}")
            remote_meta = check_html(fetch(args.url, "text/html"))
            require(remote_meta == report["metadata"], "Hosted metadata differs from local metadata")
            remote_image = fetch(remote_meta["og:image"], "image/png")
            remote_info = check_png(remote_image)
            require(remote_info["sha256"] == report["image"]["sha256"],
                    "Hosted image differs from local image")
            report["live"] = {"status": "passed", "page": args.url,
                              "image": remote_meta["og:image"], "mime": "image/png",
                              "sha256": remote_info["sha256"]}
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, zlib.error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
