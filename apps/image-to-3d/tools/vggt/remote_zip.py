#!/usr/bin/env python3
"""Read single members of a remote ZIP over HTTP range requests.

The evaluation subsets need a few hundred frames out of multi-gigabyte
archives (7-Scenes, NRGBD, Sintel). `zipfile` only needs a seekable file, so
`RangeFile` serves reads from `Range:` requests (64 KiB-aligned blocks, a
small LRU cache), and `Window` exposes a stored (uncompressed) member as a
file of its own, which is how nested archives (7-Scenes' `seq-NN.zip` inside
`<scene>.zip`) are read without downloading the outer file.

    remote_zip.py list URL [--inner MEMBER]      list members (name, size, method)

Standard library only (no third-party code), so it also runs on the host.
"""

from __future__ import annotations

import argparse
import collections
import io
import sys
import time
import urllib.request
import zipfile

BLOCK = 1 << 16


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of `url` built from HTTP range requests."""

    def __init__(self, url: str, cache_blocks: int = 256, headers: dict | None = None):
        super().__init__()
        self.url = url
        self.headers = {"User-Agent": "spl-vggt-ref"} | (headers or {})
        self.pos = 0
        self.cache: collections.OrderedDict[int, bytes] = collections.OrderedDict()
        self.cache_blocks = cache_blocks
        self.fetched = 0
        req = urllib.request.Request(url, headers=self.headers | {"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            content_range = r.headers.get("Content-Range")
            if r.status != 206 or not content_range:
                raise OSError(f"{url}: server ignored the Range header (status {r.status})")
            self.size = int(content_range.rsplit("/", 1)[1])
            self.url = r.geturl()  # follow redirects once

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        """Bytes [start, end) straight from the server, with retries."""
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    self.url, headers=self.headers | {"Range": f"bytes={start}-{end - 1}"}
                )
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) != end - start:
                    raise OSError(f"short read {len(data)} of {end - start}")
                self.fetched += len(data)
                return data
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2.0 * (attempt + 1))
        raise AssertionError

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        start, end = self.pos, self.pos + n
        if n >= 4 * BLOCK:  # large reads bypass the cache
            data = self._get(start, end)
        else:
            first, last = start // BLOCK, (end - 1) // BLOCK
            missing = [b for b in range(first, last + 1) if b not in self.cache]
            if missing:
                lo, hi = missing[0] * BLOCK, min((missing[-1] + 1) * BLOCK, self.size)
                raw = self._get(lo, hi)
                for b in range(missing[0], missing[-1] + 1):
                    self.cache[b] = raw[(b - missing[0]) * BLOCK : (b - missing[0] + 1) * BLOCK]
            parts = []
            for b in range(first, last + 1):
                self.cache.move_to_end(b)
                parts.append(self.cache[b])
            while len(self.cache) > self.cache_blocks:
                self.cache.popitem(last=False)
            blob = b"".join(parts)
            data = blob[start - first * BLOCK : end - first * BLOCK]
        self.pos = end
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


class Window(io.RawIOBase):
    """Bytes [offset, offset + size) of a seekable file, as a file."""

    def __init__(self, f, offset: int, size: int):
        super().__init__()
        self.f, self.offset, self.size, self.pos = f, offset, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        self.f.seek(self.offset + self.pos)
        data = self.f.read(n)
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def stored_member(zf: zipfile.ZipFile, raw, name: str) -> Window:
    """A stored (uncompressed) member of `zf` (opened on `raw`) as a Window."""
    info = zf.getinfo(name)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{name} is compressed (method {info.compress_type}); cannot window it")
    raw.seek(info.header_offset)
    header = raw.read(30)
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"{name}: bad local header")
    name_len = int.from_bytes(header[26:28], "little")
    extra_len = int.from_bytes(header[28:30], "little")
    return Window(raw, info.header_offset + 30 + name_len + extra_len, info.compress_size)


def open_remote(url: str) -> tuple[zipfile.ZipFile, RangeFile]:
    raw = RangeFile(url)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=BLOCK)), raw


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("url")
    ls.add_argument("--inner", help="list a stored member that is itself a zip")
    ls.add_argument("--limit", type=int, default=40)
    args = p.parse_args()
    zf, raw = open_remote(args.url)
    if args.inner:
        zf = zipfile.ZipFile(io.BufferedReader(stored_member(zf, raw, args.inner), buffer_size=BLOCK))
    infos = zf.infolist()
    print(f"{len(infos)} members, archive {raw.size} bytes")
    for info in infos[: args.limit]:
        print(f"{info.compress_type} {info.compress_size:>12} {info.file_size:>12} {info.filename}")
    print(f"fetched {raw.fetched} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
