"""Fetch the two external validation datasets into external_data/.

Downloads and unzips, using only the standard library (urllib + zipfile) so
this has no extra dependency beyond what the rest of the repo needs.

  - AdSERP (Latifzadeh, Gwizdka and Leiva, SIGIR 2025). Zenodo record
    15236546, file mouse-movement-data.zip. CC BY 4.0. Redistributable, but
    we do not commit the raw CSVs; this script is the reproducible source.
  - M4D (Iliou et al., 2021), web_bot_detection_dataset.zip, fetched from
    m4d.iti.gr. CC BY-NC-SA 4.0. NOT redistributed in this repository
    because of the NonCommercial clause; this script always pulls it fresh
    from the source rather than from a repo-hosted mirror.

Both datasets land under external_data/, which is gitignored. Existing zips
and extracted folders are left alone unless --force is passed.

Run:
    .venv/Scripts/python.exe external_validation/fetch_external_data.py
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXT_DATA_DIR = REPO_ROOT / "external_data"

ADSERP_URL = "https://zenodo.org/records/15236546/files/mouse-movement-data.zip?download=1"
ADSERP_DIR = EXT_DATA_DIR / "adserp"
ADSERP_ZIP = ADSERP_DIR / "mouse-movement-data.zip"

M4D_URL = "https://m4d.iti.gr/wp-content/uploads/2024/04/web_bot_detection_dataset.zip"
M4D_DIR = EXT_DATA_DIR / "m4d"
M4D_ZIP = M4D_DIR / "web_bot_detection_dataset.zip"


def download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"already have {dest}, skipping download (--force to redo)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "MIME-mouse/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"saved {dest} ({dest.stat().st_size} bytes)")


def unzip(zip_path: Path, extract_to: Path, force: bool) -> None:
    marker = extract_to / ".unzipped"
    if marker.exists() and not force:
        print(f"already unzipped into {extract_to}, skipping (--force to redo)")
        return
    print(f"unzipping {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")
    print(f"unzipped into {extract_to}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                     help="re-download and re-unzip even if files already exist")
    ap.add_argument("--adserp-only", action="store_true")
    ap.add_argument("--m4d-only", action="store_true")
    args = ap.parse_args()

    do_adserp = not args.m4d_only
    do_m4d = not args.adserp_only

    if do_adserp:
        print("\n=== AdSERP (CC BY 4.0, Zenodo 15236546) ===")
        download(ADSERP_URL, ADSERP_ZIP, args.force)
        unzip(ADSERP_ZIP, ADSERP_DIR, args.force)

    if do_m4d:
        print("\n=== M4D (CC BY-NC-SA 4.0, m4d.iti.gr) ===")
        download(M4D_URL, M4D_ZIP, args.force)
        unzip(M4D_ZIP, M4D_DIR, args.force)

    print("\nDone. Next steps:")
    print("  .venv/Scripts/python.exe external_validation/adserp_features.py")
    print("  .venv/Scripts/python.exe external_validation/m4d_features.py")
    print("  .venv/Scripts/python.exe external_validation/validate_adserp.py")
    print("  .venv/Scripts/python.exe external_validation/validate_m4d.py")


if __name__ == "__main__":
    main()
