# src/ingest.py

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from .config import DATA_DIR, SUPPORTED_EXTS, CHUNK_SIZE


# =============================================================================
# 1. Text cleanup (Jekyll / HTML / markdown noise)
# =============================================================================
_CLEANUP_PATTERNS = [
    re.compile(r"\{:[^}]*\}"),                 # {: .notice--warning }
    re.compile(r"\{::[^}]*\}"),                # {::comment}
    re.compile(r"<!--.*?-->", re.DOTALL),      # <!-- html comment -->
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),       # ![alt](url)
    re.compile(r"</?[a-zA-Z][^>]*>"),          # <span>, <br/>, etc.
]


def _clean_text(text: str) -> str:
    for pat in _CLEANUP_PATTERNS:
        text = pat.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text)

def _looks_like_header_reflow(s: str) -> bool:
    if not s or len(s) > 100:
        return False
    if s.endswith((".", "!", "?", ",", ";")):
        return False
    if _SECTION_NAME.match(s):
        return True
    if _ROLE_KW.search(s) and len(s) <= 80:
        return True
    if s.isupper() and 3 <= len(s) <= 80:
        return True
    return False


def _reflow_text(text: str) -> str:
    """Merge PDF fragments (one word or phrase per line) into coherent lines.

    PDF extraction often produces:
        Programming

        Python,

        SQL,

        Rust

    This collapses those into a single line: 'Programming Python, SQL, Rust'
    so the section classifier and the LLM both see a whole bullet, not a
    stream of orphan tokens.
    """
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            out.append(" ".join(buf))
            buf = []

    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue

        # Bullet marker alone on a line: start a new bullet with it.
        if re.fullmatch(r"[-*•·]", s):
            flush()
            buf = [s]
            continue

        standalone = (
            len(s) > 60
            or s.endswith((".", "!", "?", ":"))
            or _looks_like_header_reflow(s)
        )
        if standalone:
            flush()
            out.append(s)
        else:
            buf.append(s)
    flush()
    return "\n".join(out)

    
# =============================================================================
# 2. Structural classification
#
# A "header" line starts a new section. Everything between headers is the
# section body, and the body is packed into chunks at paragraph boundaries.
# This means a skills list, a role's bullets, or an entire "Education"
# section each stay together as one unit — no line is ever dropped.
# =============================================================================
_BULLET_RE = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")

_SECTION_NAME = re.compile(
    r"^(?:"
    r"summary|profile|objective|about|"
    r"experience|work\s+experience|professional\s+experience|employment|"
    r"education|academics|academic\s+background|"
    r"skills|technical\s+skills|core\s+competencies|competencies|"
    r"languages|programming\s+languages|"
    r"tools|technologies|frameworks|libraries|packages|"
    r"projects|selected\s+projects|research|research\s+experience|"
    r"publications|papers|talks|presentations|"
    r"awards|honors|grants|"
    r"certifications|licenses|"
    r"volunteer|service|leadership|"
    r"interests|activities|"
    r"contact|references"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

_ROLE_KW = re.compile(
    r"\b(Engineer|Scientist|Analyst|Manager|Developer|Researcher|"
    r"Intern|Consultant|Lead|Director|Associate|Assistant|Fellow|"
    r"Postdoc|Professor|Lecturer|Founder|Co-?founder|"
    r"University|Institute|College|Laboratory|Lab|Center|Centre|"
    r"School|Department|Group|Division)\b",
    re.IGNORECASE,
)

_SKILL_SECTION = re.compile(
    r"(skill|librar|package|framework|tool|technolog|language|competenc|"
    r"analytic|big\s*data|cloud)",
    re.IGNORECASE,
)

def _is_skill_section(header: str) -> bool:
    return bool(_SKILL_SECTION.search(header))
def _classify_line(line: str) -> str:
    """Return 'header', 'blank', or 'body'."""
    s = line.strip()
    if not s:
        return "blank"
    if _BULLET_RE.match(s):
        return "body"                            # a bullet is never a header
    if s.endswith((".", "!", "?", ",", ";")):
        return "body"
    if len(s) > 120:
        return "body"
    if _SECTION_NAME.match(s):
        return "header"
    if _ROLE_KW.search(s) and len(s) <= 100:
        return "header"
    if s.isupper() and 3 <= len(s) <= 80:
        return "header"
    return "body"


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (header, body) pairs. Consecutive empty headers merge."""
    sections: list[tuple[str, list[str]]] = []
    current_header = ""
    current_body: list[str] = []

    def flush():
        nonlocal current_header, current_body
        body = "\n".join(current_body).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        if current_header or body:
            sections.append((current_header, body))
        current_header = ""
        current_body = []

    for line in text.split("\n"):
        cls = _classify_line(line)
        if cls == "header":
            flush()
            current_header = line.strip()
        elif cls == "blank":
            if current_body and current_body[-1] != "":
                current_body.append("")
        else:
            current_body.append(line.rstrip())
    flush()

    # Merge adjacent headers that carry no body — e.g. "Data Analyst" followed
    # by "Flatiron Institute" followed by "Aug 2021 – Jan 2022".
    merged: list[tuple[str, str]] = []
    pending: list[str] = []
    for header, body in sections:
        if header and not body:
            pending.append(header)
            continue
        if pending:
            header = " | ".join(pending + ([header] if header else []))
            pending = []
        merged.append((header, body))
    if pending:
        merged.append((" | ".join(pending), ""))
    # Merge consecutive skill-like sections into one.
    merged2: list[tuple[str, str]] = []
    for header, body in merged:
        if (merged2 and _is_skill_section(merged2[-1][0])
                and _is_skill_section(header)):
            ph, pb = merged2[-1]
            merged2[-1] = (f"{ph} | {header}",
                           (pb + "\n\n" + body).strip())
        else:
            merged2.append((header, body))
    return merged2

# =============================================================================
# 3. Paragraph-aware packing
# =============================================================================
def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_oversized_paragraph(para: str, limit: int) -> list[str]:
    """Split a single paragraph at sentence boundaries; hard-split if needed."""
    sentences = re.split(r"(?<=[.!?])\s+", para)
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for s in sentences:
        if size + len(s) + 1 > limit and buf:
            out.append(" ".join(buf))
            buf, size = [], 0
        buf.append(s)
        size += len(s) + 1
    if buf:
        out.append(" ".join(buf))

    # No sentence boundaries → hard split.
    if len(out) == 1 and len(out[0]) > limit:
        big = out[0]
        return [big[i:i + limit] for i in range(0, len(big), limit)]
    return out or [para]


def _make_doc(source: str, text: str, section: str, provenance: str) -> Document:
    prefix = f"[{source} — {provenance}]" if provenance else f"[{source}]"
    return Document(
        page_content=f"{prefix}\n{text}",
        metadata={
            "source": source,
            "section": section,
            "section_type": "own",
            "provenance": provenance or source,
        },
    )


def _pack_text(text: str, source: str, section: str, provenance: str) -> list[Document]:
    """Pack text into Documents of ≤ CHUNK_SIZE, splitting only at paragraph
    boundaries. A single short section always yields exactly one chunk."""
    if not text.strip():
        return []
    if len(text) <= CHUNK_SIZE:
        return [_make_doc(source, text, section, provenance)]

    paras = _split_paragraphs(text)
    docs: list[Document] = []
    buf: list[str] = []
    size = 0

    def flush_buf():
        nonlocal buf, size
        if not buf:
            return
        docs.append(_make_doc(source, "\n\n".join(buf), section, provenance))
        buf, size = [], 0

    for p in paras:
        plen = len(p) + 2
        if plen > CHUNK_SIZE:
            flush_buf()
            for sub in _split_oversized_paragraph(p, CHUNK_SIZE):
                docs.append(_make_doc(source, sub, section, provenance))
            continue
        if size + plen > CHUNK_SIZE and buf:
            flush_buf()
        buf.append(p)
        size += plen
    flush_buf()
    return docs


# =============================================================================
# 4. Document-type chunkers
# =============================================================================
def _chunk_general(doc: Document) -> list[Document]:
    """CV, README, plain text — structural section splitting."""
    source = Path(doc.metadata.get("source", "?")).name
    sections = _split_into_sections(doc.page_content)
    out: list[Document] = []
    for header, body in sections:
        if header and body:
            text, prov = f"{header}\n\n{body}", header
        elif header:
            text, prov = header, header
        else:
            text, prov = body, source
        out.extend(_pack_text(text, source, "general", prov))
    return out


def _chunk_publication(doc: Document) -> list[Document]:
    """Publications — already section-tagged by _split_publication_pages;
    chunk each section at paragraph boundaries."""
    source = Path(doc.metadata.get("source", "?")).name
    section = doc.metadata.get("section", "general")
    prov = f"{source} — {section}"
    return _pack_text(doc.page_content, source, section, prov)


# =============================================================================
# 5. Publication section splitter (unchanged structurally)
# =============================================================================
SECTION_PATTERNS = [
    ("abstract",        r"abstract"),
    ("introduction",    r"introduction|intro"),
    ("background",      r"background|related\s+work|related\s+literature|"
                        r"prior\s+work|literature\s+review|motivation"),
    ("data",            r"data(?:\s+(?:collection|acquisition|description|"
                        r"set|source|sources|preparation))?|datasets?|"
                        r"study\s+(?:design|population)|participants|"
                        r"corpus|corpora|materials"),
    ("methods",         r"methods?|methodology|approach|materials\s+and\s+methods|"
                        r"experimental\s+(?:setup|design|section|details|"
                        r"procedure|protocol)|"
                        r"proposed\s+(?:method|approach|system|framework|model)|"
                        r"our\s+(?:method|approach|framework|model|system)|"
                        r"model\s+architecture|implementation\s+details|"
                        r"experiments?\s+and\s+methods"),
    ("results",         r"results?|findings?|experiments?|evaluation|"
                        r"empirical\s+(?:results|evaluation)|"
                        r"experimental\s+results|"
                        r"performance\s+(?:analysis|evaluation|comparison)"),
    ("discussion",      r"discussion|limitations|threats\s+to\s+validity"),
    ("conclusion",      r"conclusions?|concluding\s+remarks|"
                        r"future\s+(?:work|directions)|"
                        r"summary\s+and\s+(?:conclusions?|future)"),
    ("acknowledgments", r"acknowledg(?:e)?ments?"),
    ("references",      r"references|bibliography|works\s+cited|"
                        r"literature\s+cited|reference\s+list"),
    ("appendix",        r"appendi(?:x|ces)"),
]

_HEADING_PREFIX = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.?)?\s*"
    r"(?P<name>[A-Za-z][A-Za-z\s\-]{2,60})"
    r"(?:\s*[:.]\s*|\s+|$)"
)


def _extract_heading(line: str) -> tuple[str | None, str]:
    s = line.strip()
    if not s or len(s) > 120:
        return None, line
    m = _HEADING_PREFIX.match(s)
    if not m:
        return None, line
    candidate = m.group("name").strip().rstrip(":. ")
    for canonical, pattern in SECTION_PATTERNS:
        if re.fullmatch(pattern, candidate, re.IGNORECASE):
            return canonical, s[m.end():].strip()
    return None, line


def _split_publication_pages(pages: list[Document]) -> list[Document]:
    tagged: list[Document] = []
    current_section = "front_matter"
    buf: list[str] = []

    def flush(page_meta: dict) -> None:
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            tagged.append(Document(
                page_content=text,
                metadata={
                    **page_meta,
                    "section": current_section,
                    "section_type": "own",
                },
            ))
        buf = []

    for page in pages:
        for line in page.page_content.split("\n"):
            heading, remainder = _extract_heading(line)
            if heading is None:
                buf.append(line)
                continue
            flush(page.metadata)
            current_section = heading
            if remainder:
                buf.append(remainder)
        flush(page.metadata)
    return tagged


# =============================================================================
# 6. Public API
# =============================================================================
def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    docs: list[Document] = []
    for fp in Path(data_dir).rglob("*"):
        if not fp.is_file():
            continue
        ext = fp.suffix.lower()
        is_publication = "publications" in fp.parts
        try:
            if ext == ".pdf":
                pages = PyPDFLoader(str(fp)).load()
                for p in pages:
                    p.page_content = _reflow_text(_clean_text(p.page_content))
                else:
                    for p in pages:
                        p.metadata["section"] = "general"
                        p.metadata["section_type"] = "own"
                        docs.append(p)
            elif ext in SUPPORTED_EXTS:
                for d in TextLoader(
                    str(fp), encoding="utf-8", autodetect_encoding=True,
                ).load():
                    d.page_content = _reflow_text(_clean_text(d.page_content))
                    d.metadata["section"] = "general"
                    d.metadata["section_type"] = "own"
                    docs.append(d)
        except Exception as e:
            print(f"[ingest] Skipped {fp}: {e}")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Content-aware chunker. Never drops a line, never splits a bullet list,
    never breaks a role/entry apart from its heading."""
    out: list[Document] = []
    for doc in docs:
        section = doc.metadata.get("section", "general")
        if section == "general":
            out.extend(_chunk_general(doc))
        else:
            out.extend(_chunk_publication(doc))
    return out