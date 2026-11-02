#!/usr/bin/env python3
"""Fetch and verify the image-to-3d benchmark datasets listed in datasets.toml.

Standard library only (Python 3.11+), so it runs on the bare GPU host.

  fetch_datasets.py                 fetch every missing scene, then verify all
  fetch_datasets.py bicycle truck   only these scenes
  fetch_datasets.py --verify        re-hash every scene against the manifest
  fetch_datasets.py --print-hashes  print the tree hashes as TOML lines

fetch = "ranges" archives are never downloaded whole. The script reads the
zip's central directory with HTTP range requests, then fetches only the byte
ranges of the kept members into a sparse local file and extracts them with
zipfile (which CRC32-checks every member). The sparse file is deleted after.
fetch = "full" archives are streamed to disk while hashing (sha256), checked,
extracted (kept members only) and deleted.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import os
import re
import sys
import time
import tomllib
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "datasets.toml"
CHUNK = 1 << 20
SUMS = "SHA256SUMS"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def urlopen(req: urllib.request.Request, tries: int = 5):
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=60)
        except OSError as e:  # URLError and timeouts are OSError subclasses
            if attempt == tries - 1:
                raise
            log(f"  retry {attempt + 1} after {e}")
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def head(url: str) -> dict[str, str]:
    r = urlopen(urllib.request.Request(url, method="HEAD"))
    return {k.lower(): v for k, v in r.headers.items()}


def gcs_md5(headers: dict[str, str]) -> str | None:
    """MD5 from GCS x-goog-hash (base64) or a plain quoted ETag."""
    import base64

    for part in headers.get("x-goog-hash", "").split(","):
        part = part.strip()
        if part.startswith("md5="):
            return base64.b64decode(part[4:]).hex()
    etag = headers.get("etag", "").strip('"')
    return etag if re.fullmatch(r"[0-9a-f]{32}", etag) else None


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file over HTTP range requests."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + off
        return self.pos

    def readinto(self, b) -> int:  # type: ignore[override]
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = fetch_range(self.url, self.pos, self.pos + n)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    with urlopen(req) as r:
        if r.status != 206:
            raise RuntimeError(f"server ignored the range request ({r.status})")
        return r.read()


def download_range_into(url: str, fd: int, start: int, end: int, progress) -> None:
    """Stream [start, end) of url into fd at the same offset, retrying on errors."""
    pos = start
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-{end - 1}"})
            with urlopen(req) as r:
                if r.status != 206:
                    raise RuntimeError(f"server ignored the range request ({r.status})")
                while pos < end:
                    buf = r.read(min(CHUNK, end - pos))
                    if not buf:
                        raise OSError("short read")
                    os.pwrite(fd, buf, pos)
                    pos += len(buf)
                    progress(len(buf))
            return
        except OSError as e:
            if attempt == 5:
                raise
            log(f"  range {pos}-{end} retry {attempt + 1} after {e}")
            time.sleep(2 * (attempt + 1))


class Progress:
    def __init__(self, total: int, label: str):
        self.total, self.label, self.done, self.t0, self.last = total, label, 0, time.time(), 0.0

    def __call__(self, n: int) -> None:
        self.done += n
        now = time.time()
        if now - self.last > 10 or self.done >= self.total:
            self.last = now
            rate = self.done / max(now - self.t0, 1e-6) / 1e6
            log(f"  {self.label}: {self.done / 1e6:,.0f} / {self.total / 1e6:,.0f} MB ({rate:.1f} MB/s)")


def safe_member_path(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise RuntimeError(f"unsafe path in archive: {name!r}")
    return p


def kept(scene: dict, name: str) -> str | None:
    """Return the path relative to the scene dir if the member is kept."""
    src = scene["src"]
    if not name.startswith(src) or name.endswith("/"):
        return None
    rel = name[len(src) :]
    for k in scene["keep"]:
        if (k.endswith("/") and rel.startswith(k)) or rel == k:
            return rel
    return None


def extract(z: zipfile.ZipFile, scenes: dict[str, dict], root: Path, members) -> int:
    n = 0
    for info, sid, rel in members:
        scene = scenes[sid]
        dest = root / scene_dir(scene) / safe_member_path(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with z.open(info) as src, open(tmp, "wb") as out:  # zipfile checks CRC32 at EOF
            while buf := src.read(CHUNK):
                out.write(buf)
        os.replace(tmp, dest)
        n += 1
    return n


def scene_dir(scene: dict) -> Path:
    return Path(scene["_dataset_dir"]) / scene["_id"]


def select(z: zipfile.ZipFile, scenes: dict[str, dict]):
    out = []
    for info in z.infolist():
        safe_member_path(info.filename)
        for sid, scene in scenes.items():
            rel = kept(scene, info.filename)
            if rel is not None:
                out.append((info, sid, rel))
    return out


def fetch_archive_ranges(arc: dict, scenes: dict[str, dict], root: Path, partial: Path) -> None:
    url, size = arc["url"], arc["bytes"]
    log(f"reading central directory of {url}")
    z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url, size), 1 << 20))
    members = select(z, scenes)
    if not members:
        raise RuntimeError(f"no kept members found in {url}")
    # A member's bytes run from its local header to the next local header (or
    # the central directory), which also covers any data descriptor.
    offsets = sorted(i.header_offset for i in z.infolist())
    start_dir = z.start_dir  # type: ignore[attr-defined]
    nxt = {o: (offsets[k + 1] if k + 1 < len(offsets) else start_dir) for k, o in enumerate(offsets)}
    spans = sorted((i.header_offset, nxt[i.header_offset]) for i, _, _ in members)
    runs: list[list[int]] = []
    for a, b in spans:
        if runs and a <= runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], b)
        else:
            runs.append([a, b])
    runs.append([start_dir, size])  # central directory and end records
    total = sum(b - a for a, b in runs)
    log(f"  {len(members)} members in {len(runs)} runs, {total / 1e6:,.0f} MB of {size / 1e9:.1f} GB")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "wb") as f:
        f.truncate(size)  # sparse: only the fetched runs take disk space
    fd = os.open(partial, os.O_RDWR)
    try:
        prog = Progress(total, PurePosixPath(url).name)
        # Split long runs so a few threads keep the link busy.
        pieces = []
        for a, b in runs:
            for s in range(a, b, 64 << 20):
                pieces.append((s, min(b, s + (64 << 20))))
        with cf.ThreadPoolExecutor(4) as ex:
            for fut in [ex.submit(download_range_into, url, fd, a, b, prog) for a, b in pieces]:
                fut.result()
    finally:
        os.close(fd)
    with zipfile.ZipFile(partial) as zl:
        local = {i.filename: i for i in zl.infolist()}
        n = extract(zl, scenes, root, [(local[i.filename], sid, rel) for i, sid, rel in members])
    partial.unlink()
    log(f"  extracted {n} files")


def fetch_archive_full(arc: dict, scenes: dict[str, dict], root: Path, partial: Path) -> str:
    url, size = arc["url"], arc["bytes"]
    log(f"downloading {url}")
    partial.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    prog = Progress(size, PurePosixPath(url).name)
    with urlopen(urllib.request.Request(url)) as r, open(partial, "wb") as out:
        while buf := r.read(CHUNK):
            h.update(buf)
            out.write(buf)
            prog(len(buf))
    digest = h.hexdigest()
    if partial.stat().st_size != size:
        raise RuntimeError(f"size mismatch for {url}: {partial.stat().st_size} != {size}")
    want = arc.get("sha256", "")
    if HEX64.match(want) and want != digest:
        raise RuntimeError(f"sha256 mismatch for {url}: {digest} != {want}")
    log(f"  sha256 {digest}" + (" (matches manifest)" if want == digest else " (manifest has none)"))
    with zipfile.ZipFile(partial) as z:
        n = extract(z, scenes, root, select(z, scenes))
    partial.unlink()
    log(f"  extracted {n} files")
    return digest


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while buf := f.read(CHUNK):
            h.update(buf)
    return h.hexdigest()


def tree_listing(base: Path, subdirs_or_files: list[str]) -> str:
    files: list[Path] = []
    for k in subdirs_or_files:
        p = base / k.rstrip("/")
        if p.is_dir():
            files += [q for q in p.rglob("*") if q.is_file()]
        elif p.is_file():
            files.append(p)
        else:
            raise RuntimeError(f"missing {p}")
    rels = sorted(q.relative_to(base).as_posix() for q in files)
    return "".join(f"{file_sha256(base / r)}  {r}\n" for r in rels)


def tree_hash(listing: str) -> str:
    return hashlib.sha256(listing.encode()).hexdigest()


def load_manifest() -> tuple[dict, Path]:
    with open(MANIFEST, "rb") as f:
        m = tomllib.load(f)
    root = Path(os.path.expanduser(m["root"]))
    for sid, s in m["scenes"].items():
        s["_id"] = sid
        s["_dataset_dir"] = s["dataset"]
    return m, root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*", help="scene ids (default: all)")
    ap.add_argument("--root", help="override the manifest root")
    ap.add_argument("--verify", action="store_true", help="re-hash scenes even if present")
    ap.add_argument("--print-hashes", action="store_true", help="print sha256_tree lines for the manifest")
    args = ap.parse_args()

    m, root = load_manifest()
    if args.root:
        root = Path(args.root).expanduser()
    wanted = args.scenes or list(m["scenes"])
    unknown = [s for s in wanted if s not in m["scenes"]]
    if unknown:
        ap.error(f"unknown scenes: {unknown}")
    root.mkdir(parents=True, exist_ok=True)

    missing = [s for s in wanted if not (root / scene_dir(m["scenes"][s]) / SUMS).exists()]
    by_archive: dict[str, dict[str, dict]] = {}
    for sid in missing:
        by_archive.setdefault(m["scenes"][sid]["archive"], {})[sid] = m["scenes"][sid]

    report: dict = {"archives": {}, "scenes": {}}
    for aid, scenes in by_archive.items():
        arc = m["archives"][aid]
        hdr = head(arc["url"])
        got = int(hdr.get("content-length", -1))
        if got != arc["bytes"]:
            raise RuntimeError(f"{aid}: server size {got} != manifest {arc['bytes']}")
        md5 = gcs_md5(hdr)
        if arc.get("md5") and md5 and md5 != arc["md5"]:
            raise RuntimeError(f"{aid}: server md5 {md5} != manifest {arc['md5']}")
        partial = root / ".partial" / f"{aid}.zip"
        if arc["fetch"] == "ranges":
            fetch_archive_ranges(arc, scenes, root, partial)
        else:
            report["archives"][aid] = {"sha256": fetch_archive_full(arc, scenes, root, partial)}
        for scene in scenes.values():
            d = root / scene_dir(scene)
            (d / SUMS).write_text(tree_listing(d, scene["keep"]))

    bad = 0
    for sid in wanted:
        scene = m["scenes"][sid]
        d = root / scene_dir(scene)
        if args.verify or sid in missing:
            listing = tree_listing(d, scene["keep"])
            if listing != (d / SUMS).read_text():
                log(f"{sid}: files differ from {d / SUMS}")
                bad += 1
        else:
            listing = (d / SUMS).read_text()
        th = tree_hash(listing)
        want = scene.get("sha256_tree", "")
        status = "ok" if want == th else ("MISMATCH" if HEX64.match(want) else "unpinned")
        bad += status == "MISMATCH"
        entry = {"sha256_tree": th, "files": listing.count("\n"), "status": status}
        report["scenes"][sid] = entry
        log(f"{sid}: {entry['files']} files, tree {th[:16]}… {status}")

    (root / "fetch-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.print_hashes:
        for aid, a in report["archives"].items():
            print(f'# [archives.{aid}] sha256 = "{a["sha256"]}"')
        for sid, e in report["scenes"].items():
            print(f'# [scenes.{sid}] sha256_tree = "{e["sha256_tree"]}"')
    du = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    log(f"datasets on disk: {du / 1e9:.2f} GB under {root}; {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
