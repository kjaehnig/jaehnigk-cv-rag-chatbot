"""Prompt construction, agentic routing, and response cleaning."""
from __future__ import annotations

import re

from .config import AUTHOR_NAME


PRONOUN_RULES = f"""
PRONOUNS — mandatory:
- Always refer to {AUTHOR_NAME} using they/them/their/theirs/themself.
- Never use he/him, she/her, his, hers, himself, or herself.
- Paraphrase quoted text that uses other pronouns instead of quoting it.
"""

PROVENANCE_RULES = """
PROVENANCE — attribution:
1. Every numbered context block is scoped to ONE role, project, or paper.
2. A specific tool or method from block A must never be attributed to a
   role described only in block B.
3. A skill listed in a Skills section may be described as Karl's general
   toolkit, but the specific methods of a publication do not transfer to
   other roles.
4. ROLE-ANCHORED QUESTIONS — when asked what Karl did AT a specific
   employer/lab, answer ONLY from blocks whose provenance header names
   that employer.
"""

FORMAT_RULES = """
FORMAT — non-negotiable:
- Plain prose or a single markdown bullet list. Nothing else.
- Never output JSON, YAML, XML, or code fences.
- Never repeat the answer in multiple formats.
- Never begin with "The context...", "Based on...", or "Here are...".
"""

CONTENT_RULES = """
CONTENT:
1. "we propose", "our method" are Karl's contributions.
2. Citations like (Smith, 2020) are OTHER researchers' work.
3. Never invent metrics, datasets, venues, or collaborators.
4. Describe ONLY what the context says the project is and does.
5. Do NOT add praise, evaluation, or impact claims. Never write
   "significant", "impressive", "widely recognized", "cutting edge",
   "state of the art", "contribution to the field", or "demonstrated
   the effectiveness".
6. Do NOT add framing like "This work demonstrates" or "The project
   showcases". Start with what the project actually is, then what it
   does. Stop there.
7. Do NOT narrate what the source document says. Never write "is
   described as", "the final model is", "as mentioned in", "the
   schematic shows". State the facts directly.
8. If the context is a short project description, write a short answer.
   Match the length of the source.
"""

_SYSTEM = f"""
You answer questions about {AUTHOR_NAME}'s professional work.

{PRONOUN_RULES}
{PROVENANCE_RULES}
{FORMAT_RULES}
{CONTENT_RULES}

WHEN INFORMATION IS MISSING:
Reply with exactly: "I don't have that information in {AUTHOR_NAME}'s materials."
""".strip()


_LIST_TRIGGERS = (
    "list", "what are my", "name the", "which", "enumerate",
    "give me all", "how many", "what tools", "what skills",
    "what libraries", "what packages", "what frameworks", "what languages",
)

_BAD_PRONOUN = re.compile(
    r"\b(he|him|his|she|her|hers|himself|herself)\b", re.IGNORECASE,
)
_PRONOUN_MAP = {
    "he": "they", "him": "them", "his": "their",
    "she": "they", "her": "their", "hers": "theirs",
    "himself": "themself", "herself": "themself",
}

_FALLBACK = f"I don't have that information in {AUTHOR_NAME}'s materials."

_PROMPT_LEAK = re.compile(
    r"(The user asked for a list|Output a markdown bullet list|"
    r"every item MUST begin|One item per line|Do not output bare words|"
    r"Do not use numbers|Do not use commas|Do not skip items|"
    r"If the context contains|Here is the list of|"
    r"Context \(numbered|Remember: refer to|Give a complete, useful answer)",
    re.IGNORECASE,
)

_LEAK_MARKERS = [
    # Section headers from the system prompt
    "PRONOUNS:",
    "PROVENANCE:",
    "ROLE-ANCHORED",
    "ROLE-ANCHORED QUESTIONS",
    "FORMAT —",
    "CONTENT:",
    "CONTENT RULES",
    # Rule sentences
    "Every numbered context block",
    "Always refer to",
    "Never use he/him",
    "Never use he/ him",
    "Paraphrase quoted",
    "A skill listed in",
    "A specific tool or method",
    "answer ONLY from blocks",
    "When information is missing",
    "Reply with exactly:",
    # Prompt-form instructions
    "Context (numbered",
    "Remember: refer to",
    "Give a complete, useful answer",
    "Format: one '- item'",
    "Refer to ",
    # Old leaks already covered
    "The user asked for a list",
    "Output a markdown bullet list",
    "every item MUST begin",
    "Do not output bare words",
    "Do not use numbers",
    "Do not use commas",
    "Do not skip items",
    "If the context contains",
    "Here is the list of",
    # Anti-hallucination leakbacks
    "Citations like",
    "Never invent metrics",
    "we propose.*are Karl's",
]


_EDITORIAL_PATTERNS = [
    # Contributions / significance
    re.compile(r"[^.]*?(?:significant|impressive|cutting[- ]edge|"
               r"state[- ]of[- ]the[- ]art|widely[- ]recognized|"
               r"well[- ]recognized|important)\s+"
               r"(?:contribution|advance|work|project)[^.]*\.", re.I),
    # Field contributions
    re.compile(r"[^.]*?(?:contributes|contribution|advancement)\s+to\s+"
               r"(?:the\s+)?field\s+of\s+[^.]*\.", re.I),
    # Showcasing / demonstrating
    re.compile(r"[^.]*?(?:showcases|showcasing|demonstrates|demonstrating|"
               r"highlights|highlighting)\s+(?:the\s+)?"
               r"(?:benefits?|effectiveness|capabilities|potential|"
               r"importance)[^.]*\.", re.I),
    # Impact claims
    re.compile(r"[^.]*?(?:has been|is now|is widely|have been)\s+"
               r"(?:applied|adopted|used|recognized)[^.]*\.", re.I),
    # Cliché conclusions
    re.compile(r"[^.]*?(?:in\s+conclusion|to\s+summarize|overall,)"
               r"[^.]*\.", re.I),
]


_FIRST_PERSON_REPLACEMENTS = [
    (r"\bI'm\b", "they're"),
    (r"\bI've\b", "they've"),
    (r"\bI'll\b", "they'll"),
    (r"\bI'd\b", "they'd"),
    (r"\bI\b(?!')", "Karl"),
    (r"\bmy\b", "their"),
    (r"\bme\b", "them"),
    (r"\bmine\b", "theirs"),
    (r"\bmyself\b", "themself"),
]

_FIRST_PERSON_TOKEN = re.compile(
    r"\b(I|I'm|I've|I'll|I'd|my|me|mine|myself)\b(?!')"
)

_FIGURE_REF = re.compile(
    r"\s*(?:is |are )?(?:plotted|shown|displayed|illustrated|depicted)"
    r"\s+(?:below|above)\s+(?:in |as )?(?:figure|fig\.?)\s*\d+\.?",
    re.IGNORECASE,
)


def _to_third_person(text: str) -> str:
    """Convert first-person prose to third-person.

    Source docs are written in first person ("I decided...", "my model").
    Without conversion, direct-quote responses read as the candidate
    narrating, which is jarring in a portfolio chatbot.

    Only fires when the response is majority first-person — avoids
    munging third-person content that happens to include "I" in a quote.
    """
    if not text.strip():
        return text

    first_person_count = len(_FIRST_PERSON_TOKEN.findall(text))
    word_count = len(text.split())
    if word_count < 5 or first_person_count / word_count < 0.02:
        return text

    for pat, repl in _FIRST_PERSON_REPLACEMENTS:
        text = re.sub(pat, repl, text)

    # Strip "plotted below in figure N" — reads oddly out of context
    text = _FIGURE_REF.sub("", text)
    # Collapse double spaces left by the figure strip
    text = re.sub(r"  +", " ", text)
    return text.strip()


def _strip_editorial_drift(text: str) -> str:
    """Remove sentences with editorial / evaluative claims.

    These are phrases small models append from their training distribution
    ("This work contributes to the field...", "widely recognized...") that
    are never backed by the retrieved context. Strip them.
    """
    for pat in _EDITORIAL_PATTERNS:
        text = pat.sub("", text)
    # Collapse extra blank space left behind
    text = re.sub(r"\.\s+\.", ".", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate_at_leak(t: str) -> str:
    """Cut everything from the first leak marker onward.

    Unlike a line-filter, this removes everything after a leak — even
    the model's own closing sentences — because once the model starts
    reciting the prompt, everything after is suspect.
    """
    t = t.strip()
    for marker in _LEAK_MARKERS:
        # Split on regex so patterns like "we propose.*are Karl's" work
        m = re.search(re.escape(marker).replace(r"\.\*", ".*"), t)
        if m and m.start() > 0:
            t = t[: m.start()].rstrip()
    return t


def _dedupe_repeated_blocks(text: str) -> str:
    """Remove repeated paragraphs and bullets.

    Small models (1-2B) degenerate into loops where the same paragraph
    is emitted 2-5 times. The repetition is usually verbatim or near-
    verbatim, so normalized comparison catches it reliably.
    """
    # Split into blocks by blank lines OR bullet markers
    # (some models emit bullets without newlines between them)
    blocks = re.split(r"\n\s*\n|(?=\s-\s)", text.strip())

    seen = set()
    kept = []
    for block in blocks:
        if not block.strip():
            continue
        # Normalize: strip bullet prefix, collapse whitespace, lowercase
        norm = re.sub(r"^[\s\-*•]+", "", block.strip())
        norm = re.sub(r"\s+", " ", norm).lower().strip()

        # Also catch near-duplicates by checking first 120 chars
        key = norm[:120] if len(norm) > 120 else norm
        if key in seen:
            continue
        seen.add(key)
        kept.append(block.strip())

    return "\n\n".join(kept)


def _wants_list(q: str) -> bool:
    return any(t in q.lower() for t in _LIST_TRIGGERS)


def build_prompt(query: str, chunks: list[str], metas: list[dict] | None = None) -> str:
    metas = metas or []
    blocks = []
    for i, c in enumerate(chunks, 1):
        prov = metas[i - 1].get("provenance", "?") if i - 1 < len(metas) else "?"
        blocks.append(f"[{i}] ({prov})\n{c}")
    context = "\n\n".join(blocks) if blocks else "(no context)"

    extra = ""
    if _wants_list(query):
        extra += "\nFormat: one '- item' per line, no prose, no examples."
    extra += f"\nRefer to {AUTHOR_NAME} as they/them. Give a complete, useful answer."

    return (
        f"{_SYSTEM}{extra}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )


def _fix_pronouns(text: str, anchor: str) -> str:
    if not anchor:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in sentences:
        if anchor.lower() in s.lower():
            def _sub(m):
                r = _PRONOUN_MAP[m.group(0).lower()]
                return r.capitalize() if m.group(0)[0].isupper() else r
            s = _BAD_PRONOUN.sub(_sub, s)
        out.append(s)
    return " ".join(out)


def _clean(text: str) -> str:
    t = text.strip()

    t = _truncate_at_leak(t)

    if _FALLBACK.lower() in t.lower():
        return _FALLBACK

    lines = [ln for ln in t.split("\n") if not _PROMPT_LEAK.search(ln)]
    t = "\n".join(lines).strip()

    t = _strip_editorial_drift(t)
    t = _dedupe_repeated_blocks(t)

    if "```" in t:
        t = t.split("```")[0].strip()

    t = re.sub(r"^(answer|response|a)\s*:\s*", "", t, flags=re.IGNORECASE).strip()

    # NEW: first → third person, then pronoun fixer as backup
    t = _to_third_person(t)
    t = _fix_pronouns(t, AUTHOR_NAME)
    return t


def agentic_decide(complete_fn, query: str) -> bool:
    prompt = (
        "You route questions to a knowledge base about a data scientist. "
        "The knowledge base contains their CV, publications, and code.\n\n"
        f'Question: "{query}"\n\n'
        "Does answering this require looking up facts about THIS specific "
        "person?\n"
        "- Their publications, projects, skills, experience → SEARCH\n"
        "- Generic questions (\"what is a transformer?\") → NO_SEARCH\n\n"
        "Reply with one word: SEARCH or NO_SEARCH."
    )
    return "SEARCH" in complete_fn(prompt).upper()


def generate_response(complete_fn, query, chunks, metas=None) -> str:
    return _clean(complete_fn(build_prompt(query, chunks, metas)))