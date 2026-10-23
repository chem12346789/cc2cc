"""Remove rows with empty scf_d3bj_ene; dry-run unless --apply is supplied."""

import argparse
import csv
import io
import os
from pathlib import Path
import stat
import tempfile


PATTERN = "def2-QZVP(D)_atom-*"
COLUMN = "scf_d3bj_ene"


def clean_csv(path, apply=False):
    original = path.read_bytes()
    text = original.decode("utf-8-sig")
    lines = io.StringIO(text, newline="").readlines()
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = next(reader, None)
    if header is None or COLUMN not in header:
        print(f"SKIP {path}: missing {COLUMN} column")
        return 0
    if header.count(COLUMN) != 1:
        raise ValueError(f"{path}: duplicate {COLUMN} columns")
    column = header.index(COLUMN)
    name_column = header.index("name") if "name" in header else None
    end = reader.line_num
    kept = lines[:end]
    removed = []
    for row in reader:
        start = end
        end = reader.line_num
        if not row:
            kept.extend(lines[start:end])
            continue
        if len(row) != len(header):
            raise ValueError(f"{path}:{start + 1}: row length differs from header")
        if not row[column].strip():
            removed.append((start + 1, row))
        else:
            kept.extend(lines[start:end])

    if not removed:
        return 0
    for line, row in removed:
        name = row[name_column] if name_column is not None else ""
        print(f"{'DELETE' if apply else 'WOULD DELETE'} {path}:{line} "
              f"name={name!r} row={row!r}")

    if apply:
        updated = "".join(kept).encode("utf-8")
        if original.startswith(b"\xef\xbb\xbf"):
            updated = b"\xef\xbb\xbf" + updated
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                os.fchmod(handle.fileno(), stat.S_IMODE(path.stat().st_mode))
                handle.write(updated)
                handle.flush()
                os.fsync(handle.fileno())
            if path.read_bytes() != original:
                raise RuntimeError(f"{path}: file changed during cleanup; not replaced")
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        print(f"UPDATED {path}: removed {len(removed)} rows")
    return len(removed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1] / "validate",
        help="Validation root (default: repository validate directory)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    mode.add_argument("--apply", action="store_true", help="Rewrite affected CSV files")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error(f"validation root is not a directory: {args.root}")
    files = []
    for directory in sorted(args.root.glob(PATTERN)):
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.csv")):
            if path.is_file() and not any(
                part.is_symlink() for part in (path, *path.parents)
            ):
                files.append(path)
    removed = 0
    changed = 0
    for path in files:
        count = clean_csv(path, apply=args.apply)
        removed += count
        changed += bool(count)
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: scanned {len(files)} CSV files; "
          f"{changed} affected files; {removed} rows "
          f"{'removed' if args.apply else 'would be removed'}")


if __name__ == "__main__":
    main()
