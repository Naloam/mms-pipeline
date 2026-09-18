"""Verify and extract the official CRC-VAL-HE-7K archive into a new directory."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

from PIL import Image


EXPECTED_BYTES = 800_276_929
EXPECTED_MD5 = "2fd1651b4f94ebd818ebf90ad2b6ce06"
CLASSES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")


def digests(path: Path) -> tuple[str, str]:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def checked_members(archive: zipfile.ZipFile) -> tuple[list[zipfile.ZipInfo], dict[str, int]]:
    images: list[zipfile.ZipInfo] = []
    classes: Counter[str] = Counter()
    names: set[str] = set()
    total_bytes = 0
    for member in archive.infolist():
        name = member.filename
        parts = PurePosixPath(name).parts
        if (name.startswith("/") or "\\" in name or ".." in parts or
                member.flag_bits & 1 or stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK):
            raise ValueError(f"Unsafe ZIP member: {name!r}")
        if member.is_dir():
            continue
        if len(parts) != 3 or parts[0] != "CRC-VAL-HE-7K" or parts[1] not in CLASSES:
            raise ValueError(f"Unexpected official validation path: {name!r}")
        if not parts[2].startswith(f"{parts[1]}-") or Path(parts[2]).suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"Unexpected class/file extension: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate ZIP member: {name!r}")
        names.add(name)
        total_bytes += member.file_size
        classes[parts[1]] += 1
        images.append(member)
    if len(images) != 7180 or set(classes) != set(CLASSES) or total_bytes > 3_000_000_000:
        raise ValueError(f"Unexpected official archive inventory: {len(images)} files, {dict(classes)}")
    return images, dict(sorted(classes.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source, out = args.archive.resolve(strict=True), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    if source.stat().st_size != EXPECTED_BYTES:
        raise ValueError(f"Archive size mismatch: {source.stat().st_size}")
    md5, sha256 = digests(source)
    if md5 != EXPECTED_MD5:
        raise ValueError(f"Official archive MD5 mismatch: {md5}")
    with zipfile.ZipFile(source) as archive:
        members, by_class = checked_members(archive)
        out.mkdir(parents=True)
        archive.extractall(out)
    manifest = []
    for member in sorted(members, key=lambda item: item.filename):
        path = out / member.filename
        with Image.open(path) as image:
            if image.size != (224, 224) or image.mode != "RGB":
                raise ValueError(f"Unexpected official image format: {path} {image.mode} {image.size}")
        manifest.append({"image_id": f"CRC-VAL-HE-7K:{path.name}",
                         "class": path.parent.name, "path": str(path),
                         "sha256": digests(path)[1]})
    manifest_path = out / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as target:
        for record in manifest:
            target.write(json.dumps(record, sort_keys=True) + "\n")
    report = {
        "source_url": "https://zenodo.org/records/1214456",
        "archive_path": str(source), "archive_bytes": EXPECTED_BYTES,
        "archive_md5": md5, "archive_sha256": sha256,
        "image_count": len(manifest), "by_class": by_class,
        "format": "RGB TIFF 224x224", "manifest_sha256": digests(manifest_path)[1],
        "patient_overlap_statement": "Zenodo states 50 validation patients do not overlap training patients; no patient IDs are available here to split validation by patient.",
    }
    (out / "archive_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
