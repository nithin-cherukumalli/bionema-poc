"""Docling-based PDF parsing for the Bionema retrieval POC.

F01 produces citable document units. It deliberately preserves each source
document's locator convention instead of forcing all documents into bracketed
patent paragraph IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backend" / "ingest" / "parsed"
CACHE_FORMAT_VERSION = 3

BRACKETED_LOCATOR_RE = re.compile(r"\[\s*([0-9oOlIi|]{3,6})\s*\]", re.IGNORECASE)
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
EXAMPLE_HEADING_RE = re.compile(
    r"^(?:example|comparative example)\s+\d+[a-z]?(?:\s*[-:–]\s*.+)?$",
    re.IGNORECASE,
)
SECTION_HEADING_RE = re.compile(
    r"^(?:field|background|summary|description|detailed description|"
    r"claims?|abstract|examples?|materials and methods|results|"
    r"preparation|formulation|method|kit)\b.*$",
    re.IGNORECASE,
)
TABLE_RE = re.compile(r"\btable\s+(\d+)\b", re.IGNORECASE)
BOILERPLATE_HEADING_RE = re.compile(
    r"^(?:"
    r"WO\s*\d{4}\s*/\s*\d+|"
    r"WO\d{4}\s*/\s*\d+|"
    r"PCT\s*/\s*[A-Z]{2}\d{4}\s*/\s*\d+|"
    r"INTERNATIONAL\s*SEARCH\s*REPORT|"
    r"INTERNATIONALSEARCHREPORT|"
    r"INTERNATIONAL\s*APPLICATION\s*NO|"
    r"INTERNATIONALAPPLICATIONNO"
    r")$",
    re.IGNORECASE,
)
OCR_DIGIT_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
    }
)


@dataclass(frozen=True)
class ParsedUnit:
    doc_id: str
    doc_title: str
    source_path: str
    locator: str
    locator_type: str
    section: str
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    cache_format_version: int
    doc_id: str
    title: str
    source_path: str
    locator_type: str
    parsed_at: str
    page_range: str | None
    docling_options: dict[str, Any]
    raw_text_chars: int
    units: list[ParsedUnit]


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def manifest_documents(path: Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    manifest = load_manifest(path)
    return list(manifest.get("documents", []))


def resolve_source_path(doc_meta: dict[str, Any], project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve source file, falling back from stale extracted-text paths to PDFs."""
    manifest_file = doc_meta.get("file")
    if manifest_file:
        candidate = project_root / manifest_file
        if candidate.exists():
            return candidate

    pdf_candidate = project_root / "data" / "pdfs" / f"{doc_meta['doc_id']}.pdf"
    if pdf_candidate.exists():
        return pdf_candidate

    raise FileNotFoundError(
        f"No source file found for {doc_meta['doc_id']}; checked manifest file "
        f"{manifest_file!r} and {pdf_candidate}"
    )


def build_docling_converter(*, table_structure: bool = False) -> DocumentConverter:
    """Build a Docling converter optimized for OCR text extraction.

    The patents in /data/pdfs do not expose a usable embedded text layer, so
    Docling must OCR them. TableFormer structure extraction is disabled by
    default because F01 only needs citable text locators; table text still flows
    through OCR, while the expensive structural model can be enabled explicitly.
    """
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = table_structure
    options.force_backend_text = False
    options.accelerator_options = AcceleratorOptions(
        num_threads=max(1, min(os.cpu_count() or 4, 8)),
        device="cpu",
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        }
    )


def convert_with_docling(
    source_path: Path,
    page_range: tuple[int, int] | None = None,
    *,
    table_structure: bool = False,
) -> str:
    """Convert a source PDF to markdown using Docling.

    The project requires Docling for parsing. These patent PDFs are scanned in
    practice, so OCR is left enabled by default.
    """
    converter = build_docling_converter(table_structure=table_structure)
    kwargs: dict[str, Any] = {}
    if page_range is not None:
        kwargs["page_range"] = page_range
    result = converter.convert(source_path, **kwargs)
    return result.document.export_to_markdown()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_heading(text: str) -> str:
    cleaned = normalize_whitespace(text.strip("# -*:\t"))
    return cleaned.upper() if cleaned else "UNTITLED SECTION"


def likely_heading(line: str) -> str | None:
    cleaned = normalize_whitespace(line)
    if not cleaned:
        return None

    compact_cleaned = re.sub(r"\s+", "", cleaned)
    if BOILERPLATE_HEADING_RE.match(cleaned) or BOILERPLATE_HEADING_RE.match(compact_cleaned):
        return None

    markdown_match = MARKDOWN_HEADING_RE.match(line)
    if markdown_match:
        heading = normalize_heading(markdown_match.group(1))
        compact_heading = re.sub(r"\s+", "", heading)
        if BOILERPLATE_HEADING_RE.match(heading) or BOILERPLATE_HEADING_RE.match(compact_heading):
            return None
        return heading

    stripped = cleaned.strip(".")
    if EXAMPLE_HEADING_RE.match(stripped) or SECTION_HEADING_RE.match(stripped):
        return normalize_heading(stripped)

    words = stripped.split()
    alpha = re.sub(r"[^A-Za-z]", "", stripped)
    if 1 <= len(words) <= 12 and alpha and stripped.upper() == stripped:
        return normalize_heading(stripped)

    return None


def section_for_offset(text: str, offset: int) -> str:
    section = "DOCUMENT"
    for line in text[:offset].splitlines():
        heading = likely_heading(line)
        if heading:
            section = heading
    return section


def normalize_bracketed_locator(raw: str) -> str | None:
    digits = raw.translate(OCR_DIGIT_TRANSLATION)
    if not digits.isdigit():
        return None
    value = int(digits)
    if value < 1 or value > 999:
        return None
    return f"[{value:04d}]"


def bracketed_locator_matches(text: str) -> list[tuple[re.Match[str], str]]:
    matches: list[tuple[re.Match[str], str]] = []
    seen_offsets: set[int] = set()
    for match in BRACKETED_LOCATOR_RE.finditer(text):
        locator = normalize_bracketed_locator(match.group(1))
        if locator is None:
            continue
        if match.start() in seen_offsets:
            continue
        seen_offsets.add(match.start())
        matches.append((match, locator))
    return matches


def split_explicit_bracketed(
    *,
    text: str,
    doc_meta: dict[str, Any],
    source_path: Path,
) -> list[ParsedUnit]:
    matches = bracketed_locator_matches(text)
    units: list[ParsedUnit] = []
    seen_locators: set[str] = set()
    for index, (match, locator) in enumerate(matches):
        if locator in seen_locators:
            continue
        seen_locators.add(locator)
        start = match.end()
        end = matches[index + 1][0].start() if index + 1 < len(matches) else len(text)
        body = normalize_whitespace(text[start:end])
        if not body:
            continue
        units.append(
            ParsedUnit(
                doc_id=doc_meta["doc_id"],
                doc_title=doc_meta["title"],
                source_path=str(source_path),
                locator=locator,
                locator_type=doc_meta["paragraph_numbering"],
                section=section_for_offset(text, match.start()),
                text=body,
            )
        )
    return units


def split_section_headings(
    *,
    text: str,
    doc_meta: dict[str, Any],
    source_path: Path,
) -> list[ParsedUnit]:
    units: list[ParsedUnit] = []
    current_heading = "DOCUMENT"
    buffer: list[str] = []
    locator_counts: dict[str, int] = {}

    def unique_locator(heading: str) -> str:
        base = f"SECTION: {heading}"
        locator_counts[base] = locator_counts.get(base, 0) + 1
        if locator_counts[base] == 1:
            return base
        return f"{base} #{locator_counts[base]}"

    def flush() -> None:
        body = normalize_whitespace("\n".join(buffer))
        if not body:
            return
        units.append(
            ParsedUnit(
                doc_id=doc_meta["doc_id"],
                doc_title=doc_meta["title"],
                source_path=str(source_path),
                locator=unique_locator(current_heading),
                locator_type=doc_meta["paragraph_numbering"],
                section=current_heading,
                text=body,
            )
        )

    for line in text.splitlines():
        heading = likely_heading(line)
        if heading:
            flush()
            current_heading = heading
            buffer = []
            continue
        buffer.append(line)
    flush()

    return units


def parse_document(
    doc_meta: dict[str, Any],
    *,
    page_range: tuple[int, int] | None = None,
    table_structure: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> ParsedDocument:
    source_path = resolve_source_path(doc_meta, project_root)
    raw_text = convert_with_docling(
        source_path,
        page_range=page_range,
        table_structure=table_structure,
    )
    numbering = doc_meta["paragraph_numbering"]

    if numbering == "explicit_bracketed":
        units = split_explicit_bracketed(
            text=raw_text,
            doc_meta=doc_meta,
            source_path=source_path,
        )
    elif numbering == "section_and_example_headings":
        units = split_section_headings(
            text=raw_text,
            doc_meta=doc_meta,
            source_path=source_path,
        )
    else:
        raise ValueError(f"Unsupported paragraph_numbering: {numbering}")

    return ParsedDocument(
        cache_format_version=CACHE_FORMAT_VERSION,
        doc_id=doc_meta["doc_id"],
        title=doc_meta["title"],
        source_path=str(source_path),
        locator_type=numbering,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        page_range=f"{page_range[0]}:{page_range[1]}" if page_range else None,
        docling_options={
            "ocr": True,
            "table_structure": table_structure,
            "max_threads": max(1, min(os.cpu_count() or 4, 8)),
        },
        raw_text_chars=len(raw_text),
        units=units,
    )


def write_parsed_document(parsed: ParsedDocument, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{parsed.doc_id}.parsed.json"
    payload = asdict(parsed)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return output_path


def parsed_document_from_dict(payload: dict[str, Any]) -> ParsedDocument:
    units = [ParsedUnit(**unit) for unit in payload["units"]]
    return ParsedDocument(
        cache_format_version=payload["cache_format_version"],
        doc_id=payload["doc_id"],
        title=payload["title"],
        source_path=payload["source_path"],
        locator_type=payload["locator_type"],
        parsed_at=payload["parsed_at"],
        page_range=payload.get("page_range"),
        docling_options=payload.get("docling_options", {}),
        raw_text_chars=payload["raw_text_chars"],
        units=units,
    )


def read_cached_parsed_document(
    doc_id: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    page_range: tuple[int, int] | None = None,
    table_structure: bool = False,
) -> ParsedDocument | None:
    output_path = output_dir / f"{doc_id}.parsed.json"
    if not output_path.exists():
        return None
    with output_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("cache_format_version") != CACHE_FORMAT_VERSION:
        return None
    expected_page_range = f"{page_range[0]}:{page_range[1]}" if page_range else None
    if payload.get("page_range") != expected_page_range:
        return None
    if payload.get("docling_options", {}).get("table_structure") != table_structure:
        return None
    return parsed_document_from_dict(payload)


def parse_page_range(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    start_text, end_text = value.split(":", 1)
    return int(start_text), int(end_text)


def select_documents(
    docs: Iterable[dict[str, Any]],
    *,
    doc_id: str | None = None,
) -> list[dict[str, Any]]:
    selected = [doc for doc in docs if doc_id is None or doc["doc_id"] == doc_id]
    if doc_id and not selected:
        raise ValueError(f"doc_id not found in manifest: {doc_id}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse Bionema source PDFs with Docling.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--doc-id", help="Parse only one manifest document.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Re-run Docling even when cache exists.")
    parser.add_argument(
        "--table-structure",
        action="store_true",
        help="Enable Docling TableFormer structure extraction. Slower; OCR text extraction is default.",
    )
    parser.add_argument(
        "--page-range",
        help="Optional 1-based inclusive page range for debugging, e.g. 1:5.",
    )
    args = parser.parse_args()

    docs = select_documents(manifest_documents(args.manifest), doc_id=args.doc_id)
    page_range = parse_page_range(args.page_range)
    for doc_meta in docs:
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
        print(
            f"[ok] {parsed.doc_id}: {len(parsed.units)} locators, "
            f"{parsed.raw_text_chars} raw chars, source={source} -> {output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
