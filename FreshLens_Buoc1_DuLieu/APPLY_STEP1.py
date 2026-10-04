r"""Apply the step-1 patch to the uploaded FreshLens version, with backups.

Run from the user's project root:
    .venv\Scripts\python.exe FreshLens_Buoc1_DuLieu\APPLY_STEP1.py --project .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


def digest(path: Path) -> str:
    # Editors may change UTF-8 BOM / Windows line endings without code edits.
    content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def apply_patch(project: Path, *, check_only: bool = False) -> int:
    project = project.resolve()
    package = Path(__file__).resolve().parent
    if not (project / "src" / "train.py").is_file() or not (project / "scripts" / "db_manager.py").is_file():
        raise ValueError("Sai thu muc du an. Chon FreshLens co src/train.py va scripts/db_manager.py.")
    spec = json.loads((package / "PATCH_MANIFEST.json").read_text(encoding="utf-8"))
    pending = []
    conflicts = []
    for entry in spec["files"]:
        relative = Path(entry["path"])
        source = package / "payload" / relative
        target = project / relative
        if not target.resolve().is_relative_to(project) or not source.resolve().is_relative_to(package / "payload"):
            raise ValueError("Duong dan trong goi cap nhat khong hop le.")
        if digest(source) != entry["new_text_sha256"]:
            raise ValueError(f"Goi cap nhat khong toan ven: {relative}")
        current = digest(target) if target.is_file() else None
        if current == entry["new_text_sha256"]:
            continue
        if current != entry["base_text_sha256"]:
            conflicts.append(str(relative))
        else:
            pending.append((relative, source, target))
    if conflicts:
        print("[DUNG] Cac file sau khac ban RAR da duoc review:")
        for path in conflicts:
            print(f"  {path}")
        print("Chua thay doi file nao. Gui lai thong bao nay de ghep dung phien ban code.")
        return 2
    if check_only:
        print(f"[OK] Kiem tra phien ban: {len(pending)} file can cap nhat.")
        return 0
    if not pending:
        print("[OK] Ban sua buoc 1 da duoc ap dung day du.")
        return 0
    backup = project / "backups" / ("step1_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    backup.mkdir(parents=True)
    # Save all originals before replacing any code file.
    for relative, source, target in pending:
        if target.exists():
            old = backup / relative
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, old)
    written = []
    try:
        for relative, source, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
            try:
                shutil.copy2(source, temporary)
                temporary.replace(target)
                written.append((relative, target))
            finally:
                temporary.unlink(missing_ok=True)
    except Exception:
        for relative, target in reversed(written):
            old = backup / relative
            if old.exists():
                shutil.copy2(old, target)
            else:
                target.unlink(missing_ok=True)
        raise
    (backup / "BACKUP_INFO.json").write_text(json.dumps({"patch": "step1_data", "changed_files": [str(item[0]) for item in pending]}, indent=2), encoding="utf-8")
    print(f"[OK] Da cap nhat {len(pending)} file.")
    print(f"[BACKUP] {backup}")
    print("Buoc tiep: cai requirements-db.txt, sau do chay lenh --stats trong huong dan.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ap dung ban sua du lieu FreshLens - Buoc 1")
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true", help="Chi kiem tra phien ban, chua sua file")
    args = parser.parse_args()
    raise SystemExit(apply_patch(args.project, check_only=args.check))


if __name__ == "__main__":
    main()
