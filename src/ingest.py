"""Content-aware document parsing and chunking."""
from __future__ import annotations

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from .config import DATA_DIR, SUPPORTED_EXTS, CHUNK_SIZE


_CLEANUP_PATTERNS = [
    re.compile(r"\{:[^}]*\}"),
    re.compile(r"\{::[^}]*\}"),
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),
    re.compile(r"</?[a-zA-Z][^>]*>"),
]


def _clean_text(text: str) -> str:
    for pat in _CLEANUP_PATTERNS:
        text = pat.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text)


_BULLET_RE = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")

_SECTION_NAME = re.compile(
    r"^(?:summary|profile|objective|about|"
    r"experience|work\s+experience|professional\s+experience|employment|"
    r"education|academics|academic\s+background|"
    r"skills|technical\s+skills|core\s+competencies|competencies|"
    r"languages|programming\s+languages|"
    r"tools|technologies|frameworks|libraries|packages|"
    r"projects|selected\s+projects|research|research\s+experience|"
    r"publications|papers|talks|presentations|"
    r"awards|honors|grants|certifications|licenses|"
    r"volunteer|service|leadership|interests|activities|"
    r"contact|references)\s*:?\s*$",
    re.IGNORECASE,
)

_ROLE_KW = re.compile(
    r"\b(Engineer|Scientist|Analyst|Manager|Developer|Researcher|"
    r"Intern|Consultant|Lead|Director|Associate|Assistant|Fellow|"
    r"Postdoc|Professor|Lecturer|Founder|"
    r"University|Institute|College|Laboratory|Lab|Center|Centre|"
    r"School|Department|Group|Division|"
    r"Skill|Librar|Package|Technolog|Tool|Framework|Language|"
    r"Competenc|Proficienc)\b",
    re.IGNORECASE,
)


def _classify_line(line: str) -> str:
    s = line.strip()
    if not s:
        return "blank"
    if _BULLET_RE.match(s):
        return "body"
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
    sections, header, body = [], "", []

    def flush():
        nonlocal header, body
        b = "\n".join(body).strip()
        b = re.sub(r"\n{3,}", "\n\n", b)
        if header or b:
            sections.append((header, b))
        header, body = "", []

    for line in text.split("\n"):
        cls = _classify_line(line)
        if cls == "header":
            flush()
            header = line.strip()
        elif cls == "blank":
            if body and body[-1] != "":
                body.append("")
        else:
            body.append(line.rstrip())
    flush()

    merged, pending = [], []
    for h, b in sections:
        if h and not b:
            pending.append(h)
            continue
        if pending:
            h = " | ".join(pending + ([h] if h else []))
            pending = []
        merged.append((h, b))
    if pending:
        merged.append((" | ".join(pending), ""))

    _SKILL = re.compile(r"(skill|librar|package|framework|tool|technolog|language)", re.I)
    merged2 = []
    for h, b in merged:
        if merged2 and _SKILL.search(merged2[-1][0]) and _SKILL.search(h):
            ph, pb = merged2[-1]
            merged2[-1] = (f"{ph} | {h}", (pb + "\n\n" + b).strip())
        else:
            merged2.append((h, b))
    return merged2


def _pack(text, source, section, prov):
    if not text.strip():
        return []
    if len(text) <= CHUNK_SIZE:
        return [Document(page_content=f"[{source} — {prov}]\n{text}",
                         metadata={"source": source, "section": section,
                                   "section_type": "own",
                                   "provenance": prov or source})]

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    docs, buf, size = [], [], 0

    def flush_buf():
        nonlocal buf, size
        if buf:
            body = "\n\n".join(buf)
            docs.append(Document(
                page_content=f"[{source} — {prov}]\n{body}",
                metadata={"source": source, "section": section,
                          "section_type": "own", "provenance": prov or source},
            ))
        buf, size = [], 0

    for p in paras:
        plen = len(p) + 2
        if plen > CHUNK_SIZE:
            flush_buf()
            for i in range(0, len(p), CHUNK_SIZE):
                docs.append(Document(
                    page_content=f"[{source} — {prov}]\n{p[i:i+CHUNK_SIZE]}",
                    metadata={"source": source, "section": section,
                              "section_type": "own", "provenance": prov or source},
                ))
            continue
        if size + plen > CHUNK_SIZE and buf:
            flush_buf()
        buf.append(p)
        size += plen
    flush_buf()
    return docs


def _chunk_general(doc: Document) -> list[Document]:
    source = Path(doc.metadata.get("source", "?")).name
    out = []
    for header, body in _split_into_sections(doc.page_content):
        if header and body:
            text, prov = f"{header}\n\n{body}", header
        elif header:
            text, prov = header, header
        else:
            text, prov = body, source
        out.extend(_pack(text, source, "general", prov))
    return out


def _chunk_publication(doc: Document) -> list[Document]:
    source = Path(doc.metadata.get("source", "?")).name
    section = doc.metadata.get("section", "general")
    prov = f"{source} — {section}"
    return _pack(doc.page_content, source, section, prov)


_SECTION_PATTERNS = [
    ("abstract",        r"abstract"),
    ("introduction",    r"introduction|intro"),
    ("background",      r"background|related\s+work|related\s+literature|"
                        r"prior\s+work|literature\s+review|motivation"),
    ("data",            r"data(?:\s+(?:collection|acquisition|description|"
                        r"set|source|sources|preparation))?|datasets?|"
                        r"corpus|corpora|materials"),
    ("methods",         r"methods?|methodology|approach|materials\s+and\s+methods|"
                        r"proposed\s+(?:method|approach|system|framework|model)|"
                        r"model\s+architecture|implementation\s+details"),
    ("results",         r"results?|findings?|experiments?|evaluation|"
                        r"empirical\s+(?:results|evaluation)"),
    ("discussion",      r"discussion|limitations"),
    ("conclusion",      r"conclusions?|concluding\s+remarks|"
                        r"future\s+(?:work|directions)"),
    ("acknowledgments", r"acknowledg(?:e)?ments?"),
    ("references",      r"references|bibliography|works\s+cited"),
    ("appendix",        r"appendi(?:x|ces)"),
]

_HEADING_PREFIX = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.?)?\s*"
    r"(?P<name>[A-Za-z][A-Za-z\s\-]{2,60})"
    r"(?:\s*[:.]\s*|\s+|$)"
)


def _extract_heading(line: str):
    s = line.strip()
    if not s or len(s) > 120:
        return None, line
    m = _HEADING_PREFIX.match(s)
    if not m:
        return None, line
    cand = m.group("name").strip().rstrip(":. ")
    for canon, pat in _SECTION_PATTERNS:
        if re.fullmatch(pat, cand, re.IGNORECASE):
            return canon, s[m.end():].strip()
    return None, line


def _split_publication_pages(pages: list[Document]) -> list[Document]:
    tagged, current, buf = [], "front_matter", []

    def flush(meta):
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            tagged.append(Document(
                page_content=text,
                metadata={**meta, "section": current, "section_type": "own"},
            ))
        buf = []

    for page in pages:
        for line in page.page_content.split("\n"):
            h, rem = _extract_heading(line)
            if h is None:
                buf.append(line)
                continue
            flush(page.metadata)
            current = h
            if rem:
                buf.append(rem)
        flush(page.metadata)
    return tagged


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    docs = []
    for fp in Path(data_dir).rglob("*"):
        if not fp.is_file():
            continue
        ext = fp.suffix.lower()
        is_pub = "publications" in fp.parts
        try:
            if ext == ".pdf":
                pages = PyPDFLoader(str(fp)).load()
                for p in pages:
                    p.page_content = _clean_text(p.page_content)
                if is_pub:
                    docs.extend(_split_publication_pages(pages))
                else:
                    for p in pages:
                        p.metadata["section"] = "general"
                        p.metadata["section_type"] = "own"
                        docs.append(p)
            elif ext in SUPPORTED_EXTS:
                for d in TextLoader(str(fp), encoding="utf-8",
                                    autodetect_encoding=True).load():
                    d.page_content = _clean_text(d.page_content)
                    d.metadata["section"] = "general"
                    d.metadata["section_type"] = "own"
                    docs.append(d)
        except Exception as e:
            print(f"[ingest] Skipped {fp}: {e}")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    out = []
    for doc in docs:
        section = doc.metadata.get("section", "general")
        if section == "general":
            out.extend(_chunk_general(doc))
        else:
            out.extend(_chunk_publication(doc))
    return out