"""Verification command for F01 Docling parsing."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.ingest.parse import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    ParsedDocument,
    manifest_documents,
    parse_document,
    parse_page_range,
    read_cached_parsed_document,
    resolve_source_path,
    write_parsed_document,
)


TABLE_RE = re.compile(r"\btable\s+(\d+)\b", re.IGNORECASE)


def document_for_doc_arg(doc_arg: str | None, docs: list[dict]) -> dict | None:
    if not doc_arg:
        return None

    doc_path = (PROJECT_ROOT / doc_arg).resolve()
    if doc_path.name.endswith(".pdf"):
        for doc in docs:
            if doc_path.name == f"{doc['doc_id']}.pdf":
                return doc

    for doc in docs:
        try:
            if resolve_source_path(doc).resolve() == doc_path:
                return doc
        except FileNotFoundError:
            pass

    # Backward-compatible support for the original F01 command, whose manifest
    # path pointed at extracted text that has since been replaced by PDFs.
    for doc in docs:
        manifest_file = doc.get("file")
        if manifest_file and (PROJECT_ROOT / manifest_file).resolve() == doc_path:
            return doc

    return None


def validate_unique_locators(parsed: ParsedDocument) -> list[str]:
    counts = Counter(unit.locator for unit in parsed.units)
    duplicates = [locator for locator, count in counts.items() if count > 1]
    if duplicates:
        return [f"{parsed.doc_id}: duplicate locators found: {duplicates[:10]}"]
    return []


def validate_doc(
    parsed: ParsedDocument,
    doc_meta: dict,
    expect_paragraphs: int | None,
    *,
    partial_parse: bool,
) -> list[str]:
    errors: list[str] = []
    if not parsed.units:
        errors.append(f"{parsed.doc_id}: no parsed locators")

    expected_type = doc_meta["paragraph_numbering"]
    bad_types = [unit.locator_type for unit in parsed.units if unit.locator_type != expected_type]
    if bad_types:
        errors.append(f"{parsed.doc_id}: found locator_type values other than {expected_type}")

    if expected_type == "explicit_bracketed":
        bracketed = [unit for unit in parsed.units if re.fullmatch(r"\[\d{4}\]", unit.locator)]
        if len(bracketed) != len(parsed.units):
            errors.append(f"{parsed.doc_id}: not all locators are [NNNN] bracketed IDs")
        if expect_paragraphs is not None and len(bracketed) < expect_paragraphs:
            errors.append(
                f"{parsed.doc_id}: expected at least {expect_paragraphs} bracketed paragraphs, "
                f"found {len(bracketed)}"
            )
        if not partial_parse:
            joined = "\n".join(unit.text for unit in parsed.units)
            found_tables = {int(value) for value in TABLE_RE.findall(joined)}
            missing_tables = [number for number in range(1, 6) if number not in found_tables]
            if missing_tables:
                errors.append(f"{parsed.doc_id}: missing Table locator text for tables {missing_tables}")

    if expected_type == "section_and_example_headings":
        section_units = [unit for unit in parsed.units if unit.locator.startswith("SECTION: ")]
        if len(section_units) != len(parsed.units):
            errors.append(f"{parsed.doc_id}: not all locators use SECTION: heading format")
        example_units = [unit for unit in parsed.units if "EXAMPLE" in unit.locator.upper()]
        if not example_units:
            errors.append(f"{parsed.doc_id}: no EXAMPLE-derived section locators found")

    errors.extend(validate_unique_locators(parsed))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify F01 Docling parse output.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--doc",
        help="Optional source path to verify. Supports the original data/patent_raw.txt alias.",
    )
    parser.add_argument("--doc-id", help="Optional manifest doc_id to verify.")
    parser.add_argument("--all", action="store_true", help="Verify every manifest document.")
    parser.add_argument("--expect-paragraphs", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--page-range", help="Optional debug page range, e.g. 1:5.")
    parser.add_argument("--force", action="store_true", help="Re-run Docling even when cache exists.")
    parser.add_argument(
        "--table-structure",
        action="store_true",
        help="Enable Docling TableFormer structure extraction. Slower; OCR text extraction is default.",
    )
    args = parser.parse_args()

    docs = manifest_documents(args.manifest)
    selected: list[dict]
    if args.all:
        selected = docs
    elif args.doc_id:
        selected = [doc for doc in docs if doc["doc_id"] == args.doc_id]
        if not selected:
            print(f"[FAIL] doc_id not found: {args.doc_id}", file=sys.stderr)
            return 1
    elif args.doc:
        doc = document_for_doc_arg(args.doc, docs)
        if doc is None:
            print(f"[FAIL] doc not found in manifest/PDF fallback: {args.doc}", file=sys.stderr)
            return 1
        selected = [doc]
    else:
        selected = docs

    page_range = parse_page_range(args.page_range)
    partial_parse = page_range is not None
    all_errors: list[str] = []
    for doc_meta in selected:
        parsed = None if args.force else read_cached_parsed_document(
            doc_meta["doc_id"],
            args.output_dir,
            page_range=page_range,
            table_structure=args.table_structure,
        )
        source = "cache"
        if parsed is None:
            parsed = parse_document(
                doc_meta,
                page_range=page_range,
                table_structure=args.table_structure,
            )
            source = "docling"
        output_path = write_parsed_document(parsed, args.output_dir)
        expect = args.expect_paragraphs if len(selected) == 1 else None
        errors = validate_doc(parsed, doc_meta, expect, partial_parse=partial_parse)
        if errors:
            all_errors.extend(errors)
            print(f"[FAIL] {parsed.doc_id}: wrote {output_path}")
        else:
            print(
                f"[ok] {parsed.doc_id}: {len(parsed.units)} locators, "
                f"locator_type={parsed.locator_type}, "
                f"page_range={parsed.page_range or 'full'}, source={source}, wrote {output_path}"
            )

    if all_errors:
        for error in all_errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
