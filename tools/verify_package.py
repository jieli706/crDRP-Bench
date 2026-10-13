"""Verify snapshot integrity without running the historical analyses."""

import ast
import csv
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{50,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{32,}"),
]


def digest(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        return h.hexdigest()


def resolve_local(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path outside repository: {relative}")
    return path


def check():
    errors = []
    status = json.loads((ROOT / "release_status.json").read_text(encoding="utf-8"))
    recorded = {}
    for line in (ROOT / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if name in recorded:
            errors.append(f"Duplicate manifest path: {name}")
        recorded[name] = expected
        path = resolve_local(name)
        if not path.is_file():
            errors.append(f"Missing file: {name}")
        elif digest(path) != expected:
            errors.append(f"Checksum mismatch: {name}")
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.relative_to(ROOT).parts]
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        if name != "manifest.sha256" and name not in recorded:
            errors.append(f"Unrecorded file: {name}")
        if "__pycache__" in path.parts or path.suffix.lower() in (".pyc", ".pyo", ".log"):
            errors.append(f"Transient file: {name}")
        if path.stat().st_size > 100 * 1024 * 1024:
            errors.append(f"File exceeds ordinary GitHub limit: {name}")
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=name)
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f"Python syntax: {name}: {exc}")
        if path.suffix.lower() in (".py", ".r", ".sh", ".ps1", ".md", ".txt", ".json", ".yml", ".cff"):
            content = path.read_text(encoding="utf-8-sig", errors="replace")
            if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                errors.append(f"Credential pattern detected: {name}")
        if path.suffix == ".md" and (not name.startswith("legacy/") or name in ("legacy/README.md", "legacy/revision_20260930/README.md")):
            content = path.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
                target = target.strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or not parsed.path:
                    continue
                linked = (path.parent / unquote(parsed.path)).resolve()
                if not linked.is_relative_to(ROOT) or not linked.exists():
                    errors.append(f"Broken local link: {name}: {target}")

    tables_dir = ROOT / "out/revision/tables"
    with (tables_dir / "table_index.csv").open(encoding="utf-8", newline="") as stream:
        tables = list(csv.DictReader(stream))
    if len(tables) != 62:
        errors.append(f"Expected 62 table exports, found {len(tables)}")
    for row in tables:
        path = resolve_local("out/revision/tables/" + row["csv_file"])
        with path.open(encoding="utf-8", newline="") as stream:
            values = list(csv.reader(stream))
        if len(values) - 1 != int(row["data_rows"]):
            errors.append(f"Table row count mismatch: {row['sheet']}")
        if any(len(r) != int(row["columns"]) for r in values):
            errors.append(f"Table column count mismatch: {row['sheet']}")
        if digest(path) != row["sha256"]:
            errors.append(f"Table index checksum mismatch: {row['sheet']}")

    workbook = ROOT / "data/Supplementary_Tables_S1-S15_REV7.xlsx"
    with zipfile.ZipFile(workbook) as archive:
        tree = ET.fromstring(archive.read("xl/workbook.xml"))
        names = [s.attrib["name"] for s in tree.findall("{*}sheets/{*}sheet")]
    if len(names) != 63 or set(names) - {"README"} != {x["sheet"] for x in tables}:
        errors.append("Workbook sheet coverage does not match CSV exports")

    with (ROOT / "docs/figure_manifest.csv").open(encoding="utf-8", newline="") as stream:
        figures = list(csv.DictReader(stream))
    expected_figures = {"Figure" + str(n) for n in range(1, 7)} | {"FigureS" + str(n) for n in range(1, 19)}
    if len(figures) != 24 or {f["figure"] for f in figures} != expected_figures:
        errors.append("Figure coverage does not match 6 main and 18 supplementary figures")
    for row in figures:
        png = resolve_local(row["png"])
        pdf = resolve_local(row["pdf"])
        if digest(png) != row["embedded_rev7_png_sha256"]:
            errors.append(f"PNG differs from embedded REV7 image: {row['figure']}")
        if digest(pdf) != row["pdf_sha256"]:
            errors.append(f"PDF differs from the supplied version: {row['figure']}")
        if not png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            errors.append(f"Invalid PNG signature: {row['figure']}")
        if not pdf.read_bytes().startswith(b"%PDF-"):
            errors.append(f"Invalid PDF signature: {row['figure']}")

    report = {
        "file_integrity": "FAIL" if errors else "PASS",
        "files_checked": len(recorded),
        "table_exports": len(tables),
        "workbook_sheets": len(names),
        "main_figures": 6,
        "supplementary_figures": 18,
        "python_files_syntax_checked": sum(p.suffix == ".py" for p in files),
        "reproducibility_complete": status["reproducibility_complete"],
        "missing_corrected_release_components": [k for k, v in status["corrected_release_components"].items() if v != "included"],
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return bool(errors)


if __name__ == "__main__":
    try:
        sys.exit(check())
    except Exception as exc:
        print(json.dumps({"file_integrity": "FAIL", "error": str(exc)}, ensure_ascii=True, indent=2))
        sys.exit(1)
