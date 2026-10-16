# src/agent.py

import re

from .config import MAX_NEW_TOKENS, AUTHOR_NAME


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------
PRONOUN_RULES = f"""
PRONOUNS — mandatory, no exceptions:
- Always refer to Karl using they/them/their/theirs/themself.
- NEVER use he/him, she/her, his, hers, himself, or herself — not in
  prose, not in examples, not in quoted citations, not anywhere.
- This applies even if the context quotes text that uses other pronouns;
  paraphrase instead of quoting.
"""

PROVENANCE_RULES = f"""
PROVENANCE — attribution rules:

1. Every numbered context block is scoped to ONE role, project, or paper.
   The scope is shown in parentheses after the block number.

2. A specific TOOL, LIBRARY, or METHOD that appears only in block A must
   never be attributed to a role described only in block B. Example:
   if "Plotly" appears in a 2025 project block and "Flatiron" appears in
   a 2021 role block, do NOT say "Karl used Plotly at Flatiron."

3. A skill listed in a dedicated Skills section of the CV (e.g. "Python,
   Docker, scikit-learn") may be mentioned as part of Karl's general
   toolkit. But specific METHODS, MODELS, or PROJECTS described inside a
   publication or a specific role's bullet points do NOT transfer to
   other roles. "We used Bayesian hierarchical models" in a paper does
   not mean Karl used them at every job.

4. Distinguish the two cases:
   - "Did Karl use [specific tool from a specific project] at [specific
     role]?" → refuse unless a single block has both.
   - "Does Karl know [general skill]?" or "Did Karl use [general skill]
     in his work?" → answer yes if the skill appears anywhere in his
     profile, and cite where it's listed.

5. ROLE-ANCHORED QUESTIONS — hard rule:
   When the user asks what Karl did AT a specific employer, lab, school,
   or organization (e.g. "at Flatiron", "at Columbia", "at Google"), the
   answer must come ONLY from blocks whose provenance header names that
   same employer/lab/school. Do NOT pull methods, projects, publications,
   or tools from blocks with different provenance.

   Before naming any analysis, method, tool, or project in your answer,
   verify that the SAME provenance header that mentions the employer also
   mentions the item. If it doesn't, do not include it.

   Example of the failure to avoid:
     Block A: "[cv.pdf — Flatiron Institute | Data Analyst | 2021-2022]
               Built diagnostic pipelines for astronomical surveys."
     Block B: "[xray_paper.pdf — methods]
               We used Bayesian models on chest x-rays."
     WRONG answer: "At Flatiron, Karl used Bayesian models on chest x-rays."
     CORRECT answer: "At Flatiron, Karl built diagnostic pipelines for
     astronomical surveys."
"""

FORMAT_RULES = f"""
FORMAT — non-negotiable:
- Plain prose or a single markdown bullet list. Nothing else.
- NEVER output JSON, YAML, XML, or code fences.
- NEVER repeat the answer in multiple formats.
- NEVER add a summary paragraph after the answer.
- NEVER begin with "The context...", "Based on...", "Here are...",
  "There is no information...", or "Under no circumstances...".
- One answer only. Length should match depth of the question.
"""

CONTENT_RULES = f"""
CONTENT RULES:
1. Claims like "we propose", "our method" are Karl's contributions.
2. Citations like (Smith, 2020) or [12] are OTHER researchers' work.
   Never present cited work as Karl's.
3. Self-citations containing "Karl" are the candidate's prior work.
4. Never invent metrics, datasets, venues, or collaborators.
"""

_SYSTEM_RULES = f"""
You answer questions about Karl's professional work.

{PRONOUN_RULES}

{PROVENANCE_RULES}

{FORMAT_RULES}

{CONTENT_RULES}
WHEN INFORMATION IS MISSING:
Reply with exactly: "I don't have that information in Karl's materials."
""".strip()

_PROMPT_LEAK = re.compile(
    r"(The user asked for a list|"
    r"Output a markdown bullet list|"
    r"every item MUST begin|"
    r"One item per line|"
    r"Do not output bare words|"
    r"Do not use numbers|"          # NEW
    r"Do not use commas|"           # NEW
    r"\(dash \+ space\)|"           # NEW
    r"Do not skip items|"
    r"If the context contains \d+|"
    r"Here is the list of|"
    r"Context \(numbered, with provenance\)|"
    r"Remember: refer to \w+ as they/them|"
    r"Give a complete, useful answer)",
    re.IGNORECASE,
)


def _strip_prompt_leak(t: str) -> str:
    """Remove any line that looks like it was copied from the prompt itself."""
    lines = []
    for line in t.split("\n"):
        if _PROMPT_LEAK.search(line):
            continue
        lines.append(line)
    return "\n".join(lines)
# ---------------------------------------------------------------------------
# Regex helpers (defined BEFORE any function that uses them)
# ---------------------------------------------------------------------------
_LIST_TRIGGERS = (
    "list", "what are my", "what are karl", "name the",
    "which", "enumerate", "give me all",
    "how many", "what tools", "what skills", "what libraries",
    "what packages", "what frameworks", "what languages",
)

# Whole-word gendered pronouns.
_BAD_PRONOUN = re.compile(
    r"\b(he|him|his|she|her|hers|himself|herself)\b",
    re.IGNORECASE,
)

_PRONOUN_MAP = {
    "he": "they", "him": "them", "his": "their",
    "she": "they", "her": "their", "hers": "theirs",
    "himself": "themself", "herself": "themself",
}

# Meta-commentary lines the model tends to emit.
_META_LINE = re.compile(
    r"^\s*(here (are|is)|based on|the context|the provided|"
    r"i don't have|in summary|in conclusion|answer:|response:)\b",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------
def _wants_list(query: str) -> bool:
    return any(t in query.lower() for t in _LIST_TRIGGERS)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def build_prompt(query, context_chunks, context_meta=None):
    context_meta = context_meta or []
    blocks = []
    for i, chunk in enumerate(context_chunks, 1):
        prov = context_meta[i - 1].get("provenance", "unknown") \
            if i - 1 < len(context_meta) else "unknown"
        blocks.append(f"[{i}] ({prov})\n{chunk}")
    context = "\n\n".join(blocks) if blocks else "(no context)"

    extra = ""
    if _wants_list(query):
        extra += "\nFormat: one '- item' per line, no prose, no examples."

    extra += (
        f"\nRemember: refer to Karl as they/them. "
        "Give a complete, useful answer."
    )

    return (
        f"{_SYSTEM_RULES}{extra}\n\n"
        f"Context (numbered, with provenance):\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

def agentic_decide(llm, query: str) -> bool:
    prompt = (
        "You route questions to a knowledge base about a data scientist's "
        "career. The knowledge base contains: their CV, their publications, "
        "their code repositories, and their project descriptions.\n\n"
        f'Question: "{query}"\n\n'
        "Does answering this question require looking up facts about THIS "
        "specific person?\n"
        "- Anything about their publications, papers, projects, skills, "
        "experience, education, employers, or code → SEARCH\n"
        '- Generic questions ("what is a transformer?") → NO_SEARCH\n\n'
        "Reply with one word: SEARCH or NO_SEARCH."
    )
    out = llm(prompt, max_tokens=4, temperature=0.0, echo=False)
    return "SEARCH" in out["choices"][0]["text"].upper()


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def _fix_pronouns(text: str, anchor: str) -> str:
    """Rewrite gendered pronouns to they/them forms, but only in sentences
    that mention the candidate (so collaborators keep their own pronouns)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in sentences:
        if anchor.lower() in s.lower():
            def _sub(m):
                repl = _PRONOUN_MAP[m.group(0).lower()]
                return repl.capitalize() if m.group(0)[0].isupper() else repl
            s = _BAD_PRONOUN.sub(_sub, s)
        out.append(s)
    return " ".join(out)

_FALLBACK = "I don't have that information in Karl's materials."


def _clean(text: str) -> str:
    t = text.strip()

    # 0a. Strip any line that appears to be prompt-echo.
    t = _strip_prompt_leak(t).strip()

    # 0b. Hard stop on fallback.
    if _FALLBACK.lower() in t.lower():
        return _FALLBACK

    # 1. Kill anything after the first code fence.
    if "```" in t:
        t = t.split("```")[0].strip()

    # 2. Kill anything after the first meta-commentary line (if it's not
    #    the very start of the answer).
    m = _META_LINE.search(t)
    if m and m.start() > 0:
        t = t[: m.start()].strip()

    # 3. Strip leading "Answer:" / "Response:" labels.
    t = re.sub(r"^(answer|response|a)\s*:\s*", "", t, flags=re.IGNORECASE).strip()

    blocks = [b.strip() for b in t.split("\n\n") if b.strip()]
    if len(blocks) > 1:
        first = re.sub(r"\s+", " ", blocks[0]).lower()
        kept = [blocks[0]]
        for b in blocks[1:]:
            b_norm = re.sub(r"\s+", " ", b).lower()
            if b_norm == first or b_norm.startswith(first[:80]):
                break                          # exact/prefix duplicate → stop
            kept.append(b)
        t = "\n\n".join(kept)

    # 5. Fix pronouns.
    t = _fix_pronouns(t, AUTHOR_NAME)

    return t.strip()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_response(llm, query, context_chunks, context_meta=None,
                      max_tokens=MAX_NEW_TOKENS):
    out = llm(
        build_prompt(query, context_chunks, context_meta),
        max_tokens=min(max_tokens, 700),
        temperature=0.25,
        top_p=0.9,
        repeat_penalty=1.05,
        # frequency_penalty=0.0,
        # presence_penalty=0.0,
        stop=[
            "\n\n\n\n",
            "\nAnswer:", "\nQuestion:", "\nContext:",
            "\nContext (numbered", "Context (numbered",
            "To determine if", "We would need",
            "The context blocks", "Block [",
            "Would you like me", "Certainly!", "Please provide",
            "How can I help", "Is there anything else",
            "\nHere are", "\nIn summary", "\nIn conclusion",
            "\n```", "```json",
            "Under no circumstances", "If more context",
            "The user asked for a list",
            "Output a markdown bullet list",
            "every item MUST begin",
            "Do not output bare words",
            "One item per line",
            "Here is the list",
        ],
        echo=False,
    )
    return _clean(out["choices"][0]["text"])