"""Resume and verify the official CRC-VAL-HE-7K archive.

Existing segment files from the earlier 16-part downloader are reused. The
archive and all temporary files live beneath the caller's new output path.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time

import requests


URL = "https://zenodo.org/api/records/1214456/files/CRC-VAL-HE-7K.zip/content"
SIZE = 800_276_929
MD5 = "2fd1651b4f94ebd818ebf90ad2b6ce06"
SEGMENTS = 16
REQUEST_BYTES = 4 << 20


def bounds(index: int) -> tuple[int, int]:
    segment_size = SIZE // SEGMENTS
    start = index * segment_size
    stop = SIZE - 1 if index == SEGMENTS - 1 else start + segment_size - 1
    return start, stop


def fetch(index: int, output: Path, retries: int) -> int:
    start, stop = bounds(index)
    part = Path(f"{output}.part{index:02d}")
    expected = stop - start + 1
    failures = 0
    while True:
        present = part.stat().st_size if part.exists() else 0
        if present == expected:
            return expected
        if present > expected:
            raise ValueError(f"Oversized segment: {part}")
        begin = start + present
        request_stop = min(stop, begin + REQUEST_BYTES - 1)
        try:
            with requests.get(URL, headers={"Range": f"bytes={begin}-{request_stop}"},
                              stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                declared = response.headers.get("Content-Range")
                if response.status_code != 206 or declared != f"bytes {begin}-{request_stop}/{SIZE}":
                    raise ValueError(f"Unexpected range response {response.status_code}: {declared}")
                with part.open("ab") as target:
                    for block in response.iter_content(1 << 20):
                        if block:
                            target.write(block)
            if part.stat().st_size <= present:
                raise RuntimeError(f"Segment {index:02d} made no progress")
            failures = 0
        except requests.RequestException as exc:
            failures += 1
            print(f"segment {index:02d}, retry {failures}/{retries}: {exc}", flush=True)
            if failures >= retries:
                raise RuntimeError(f"Segment {index:02d} incomplete") from exc
            time.sleep(min(2 * failures, 20))


def verify(path: Path) -> str:
    if path.stat().st_size != SIZE:
        raise ValueError(f"Wrong archive size: {path.stat().st_size}")
    digest = hashlib.md5()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != MD5:
        raise ValueError(f"Official MD5 mismatch: {digest.hexdigest()}")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.workers <= SEGMENTS or args.retries < 1:
        parser.error("workers must be 1–16 and retries must be positive")
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"Verified existing {output}: md5={verify(output)}", flush=True)
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, i, output, args.retries) for i in range(SEGMENTS)]
        for future in as_completed(futures):
            future.result()
            complete = sum(Path(f"{output}.part{i:02d}").exists()
                           and Path(f"{output}.part{i:02d}").stat().st_size
                           == bounds(i)[1] - bounds(i)[0] + 1
                           for i in range(SEGMENTS))
            print(f"complete segments: {complete}/{SEGMENTS}", flush=True)
    temporary = Path(f"{output}.assembling")
    with temporary.open("wb") as target:
        for i in range(SEGMENTS):
            with Path(f"{output}.part{i:02d}").open("rb") as source:
                for block in iter(lambda: source.read(8 << 20), b""):
                    target.write(block)
    digest = verify(temporary)
    temporary.replace(output)
    print(f"Verified official archive: {output}, md5={digest}, bytes={SIZE}", flush=True)


if __name__ == "__main__":
    main()
