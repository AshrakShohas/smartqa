"""
qa_generator.py
Strategy: Gemini first → Groq fallback (no race, sequential).
Speed:     Generates in small batches of BATCH_SIZE questions,
           calling on_batch() after each so UI shows results progressively.
"""

import json
import re
import time
import random

# ── Gemini ────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("  ⚠️ google-genai not installed — run: pip install google-genai")

# ── Groq ──────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("  ⚠️ groq not installed — run: pip install groq")


GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

# Confirmed available April 2026 (from live model list)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",          # Best quality, large context
    "openai/gpt-oss-120b",              # Very strong, large
    "moonshotai/kimi-k2-instruct",      # Strong alternative
    "qwen/qwen3-32b",                   # Good quality
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Fast
    "openai/gpt-oss-20b",              # Lighter fallback
    "llama-3.1-8b-instant",            # Fastest, last resort
]

BATCH_SIZE      = 10     # questions per API call — first 10 show up fast, then next 10, etc.
MAX_CHARS_BATCH = 25000  # max content chars per prompt

QTYPE_INSTRUCTIONS = {
    "definition":  "Generate ONLY definition questions: 'What is...?', 'Define...'. Focus on terms.",
    "descriptive": "Generate ONLY descriptive questions: 'Explain...', 'Describe...', 'Discuss...'.",
    "formula":     "Generate ONLY formula/equation questions. Every answer must include a formula.",
    "figure":      "Generate ONLY diagram/figure questions: 'Draw and explain...', 'What does the figure show?'",
    "smart":       "Pick the MOST IMPORTANT questions covering key concepts and frequently tested ideas.",
    "mixed":       "Generate a balanced mix: definitions, descriptive, and formula questions.",
}


# ══════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════

def generate_questions(
    text, filename, api_key,
    difficulties, lengths,
    q_per_page, page_from, page_to, lang,
    metadata=None,
    smart_mode=False, smart_total=50,
    question_type="mixed", wants_summary=False,
    groq_api_key=None,
    race_mode=False,   # kept for API compat
    api_mode="auto",   # "auto" | "gemini" | "groq" | "race"
    on_batch=None,     # callback(batch_questions, summary_or_None) called as each batch completes
):
    """
    SPEED STRATEGY:
    - Split total questions into BATCH_SIZE chunks
    - Fire ALL chunks in parallel using ThreadPoolExecutor
    - Each finished batch is immediately sent to on_batch() so UI updates live
    - Total time ≈ time of ONE API call, not N sequential calls
    """
    import threading
    import concurrent.futures

    meta_ctx    = _build_meta_context(metadata)
    qtype_instr = QTYPE_INSTRUCTIONS.get(question_type, QTYPE_INSTRUCTIONS["mixed"])
    content     = text[:MAX_CHARS_BATCH]

    if smart_mode:
        total_needed = smart_total
    else:
        pages  = max(1, page_to - page_from + 1)
        n_diff = max(1, len(difficulties))
        total_needed = min(pages * q_per_page * n_diff, 120)

    # ── Build all batch prompts upfront ──────────────────────
    batches = []   # list of (batch_num, id_start, prompt_gemini, prompt_groq)
    generated = 0
    batch_num = 0
    while generated < total_needed:
        this_batch = min(BATCH_SIZE, total_needed - generated)
        batch_num += 1
        id_start   = generated + 1
        prompt_gemini = _build_batch_prompt(
            content=content, filename=filename, difficulties=difficulties,
            lengths=lengths, qtype_instr=qtype_instr, lang=lang, meta_ctx=meta_ctx,
            batch_size=this_batch, id_start=id_start, page_from=page_from,
            page_to=page_to, wants_summary=(wants_summary and batch_num == 1),
            already_asked=[],   # parallel batches can't know each other's Qs upfront
            for_groq=False,
        )
        prompt_groq = _build_batch_prompt(
            content=content, filename=filename, difficulties=difficulties,
            lengths=lengths, qtype_instr=qtype_instr, lang=lang, meta_ctx=meta_ctx,
            batch_size=this_batch, id_start=id_start, page_from=page_from,
            page_to=page_to, wants_summary=(wants_summary and batch_num == 1),
            already_asked=[],
            for_groq=True,
        )
        batches.append((batch_num, id_start, this_batch, prompt_gemini, prompt_groq))
        generated += this_batch

    print(f"  🚀 Firing {len(batches)} batches in PARALLEL (total ~{total_needed} questions)")

    # ── Thread-safe accumulator ───────────────────────────────
    results_lock  = threading.Lock()
    all_questions = []
    first_summary = [None]

    def run_batch(args):
        bnum, id_start, bsize, pg, pgr = args
        print(f"    📦 Batch {bnum} starting (IDs {id_start}–{id_start+bsize-1})")
        try:
            qs, summary = _call_api(api_key, groq_api_key, pg, pgr, api_mode=api_mode)
            for q in qs:
                q["source_file"] = filename
            with results_lock:
                all_questions.extend(qs)
                if summary and first_summary[0] is None:
                    first_summary[0] = summary
            print(f"    ✅ Batch {bnum} done → {len(qs)} questions")
            if on_batch and qs:
                on_batch(qs, summary if bnum == 1 else None)
            return len(qs)
        except Exception as e:
            print(f"    ❌ Batch {bnum} failed: {e}")
            return 0

    # ── Run all batches in parallel ───────────────────────────
    # Max workers: cap at 5 to avoid rate limits, min with number of batches
    max_workers = min(5, len(batches))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_batch, b) for b in batches]
        concurrent.futures.wait(futures)

    print(f"  🎉 All batches done. Total: {len(all_questions)} questions")
    return all_questions


# ══════════════════════════════════════════════════════════════
# API DISPATCH
# api_mode: "auto" | "gemini" | "groq" | "race"
# ══════════════════════════════════════════════════════════════

def _call_api(gemini_key, groq_key, prompt_gemini, prompt_groq=None, api_mode="auto"):
    """prompt_groq uses simpler schema that doesn't confuse Groq models."""
    if prompt_groq is None:
        prompt_groq = prompt_gemini  # fallback if caller passes one prompt
    has_gemini = bool(gemini_key) and GEMINI_AVAILABLE
    has_groq   = bool(groq_key)   and GROQ_AVAILABLE

    if not has_gemini and not has_groq:
        raise RuntimeError("No API keys found. Add GEMINI_API_KEY or GROQ_API_KEY to your .env file.")

    # Normalize: if user picks a mode but doesn't have that key, fall to auto
    if api_mode == "gemini" and not has_gemini:
        print("  ⚠️ Gemini-only mode but no Gemini key — switching to Groq")
        api_mode = "groq"
    if api_mode == "groq" and not has_groq:
        print("  ⚠️ Groq-only mode but no Groq key — switching to Gemini")
        api_mode = "gemini"
    if api_mode == "race" and not (has_gemini and has_groq):
        print("  ⚠️ Race mode needs both keys — switching to auto")
        api_mode = "auto"

    # ── RACE: fire both in parallel, use first winner ─────────
    if api_mode == "race":
        return _race(gemini_key, groq_key, prompt_gemini, prompt_groq)

    # ── GEMINI ONLY ───────────────────────────────────────────
    if api_mode == "gemini":
        return _try_gemini(gemini_key, prompt_gemini)

    # ── GROQ ONLY ─────────────────────────────────────────────
    if api_mode == "groq":
        return _try_groq(groq_key, prompt_groq)

    # ── AUTO: Gemini first → Groq fallback ────────────────────
    if has_gemini:
        try:
            return _try_gemini(gemini_key, prompt_gemini)
        except Exception as e:
            print(f"  ⚠️ Gemini fully failed ({e}) — falling back to Groq")
    if has_groq:
        return _try_groq(groq_key, prompt_groq)

    raise RuntimeError("Both Gemini and Groq failed. Check your API keys.")


def _try_gemini(gemini_key, prompt):
    """Try all Gemini models in order. Raises if all fail."""
    for model in GEMINI_MODELS:
        try:
            print(f"    → Gemini/{model}")
            raw = _gemini_call(gemini_key, model, prompt)
            summary, raw = _extract_summary(raw)
            questions = _sanitize_questions(_parse_json(raw))
            for q in questions:
                q["model_used"] = f"Gemini/{model}"
            print(f"    ✅ Gemini/{model} → {len(questions)} Qs")
            return questions, summary
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ["429", "quota", "RESOURCE_EXHAUSTED", "rate"]):
                print(f"    ⚠️ Gemini/{model} rate limited → next")
                time.sleep(2)
            elif any(x in msg for x in ["404", "NOT_FOUND", "not found"]):
                print(f"    ⚠️ Gemini/{model} not found → next")
            else:
                print(f"    ⚠️ Gemini/{model}: {e} → next")
    raise RuntimeError("All Gemini models failed. Rate limited or key invalid.")


def _try_groq(groq_key, prompt):
    """Try all Groq models in order. Raises if all fail."""
    live = _get_groq_models(groq_key)
    for model in live:
        try:
            print(f"    → Groq/{model}")
            raw = _groq_call(groq_key, model, prompt)
            summary, raw = _extract_summary(raw)
            questions = _sanitize_questions(_parse_json(raw))
            for q in questions:
                q["model_used"] = f"Groq/{model}"
            print(f"    ✅ Groq/{model} → {len(questions)} Qs")
            return questions, summary
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ["429", "rate_limit", "rate limit"]):
                print(f"    ⚠️ Groq/{model} rate limited → next")
                time.sleep(2)
            elif any(x in msg for x in ["404", "not found", "model_not_found", "decommissioned"]):
                print(f"    ⚠️ Groq/{model} not found → next")
            else:
                print(f"    ⚠️ Groq/{model}: {e} → next")
    raise RuntimeError("All Groq models failed. Rate limited or key invalid.")


def _race(gemini_key, groq_key, prompt_gemini, prompt_groq):
    """Fire Gemini + Groq simultaneously. Return first valid result."""
    import threading
    result = {"winner": None, "errors": [], "lock": threading.Lock()}

    def run(name, fn, key, prompt):
        try:
            qs, summary = fn(key, prompt)
            with result["lock"]:
                if result["winner"] is None and qs:
                    result["winner"] = (qs, summary, name)
                    print(f"  🏁 Race won by {name}!")
        except Exception as e:
            with result["lock"]:
                result["errors"].append(f"{name}: {e}")
            print(f"  ⚠️ {name} lost race: {e}")

    threads = [
        threading.Thread(target=run, args=("Gemini", _try_gemini, gemini_key, prompt_gemini), daemon=True),
        threading.Thread(target=run, args=("Groq",   _try_groq,   groq_key,   prompt_groq),  daemon=True),
    ]
    for t in threads: t.start()

    deadline = time.time() + 90
    while time.time() < deadline:
        with result["lock"]:
            if result["winner"]: break
            if len(result["errors"]) >= 2: break
        if not any(t.is_alive() for t in threads): break
        time.sleep(0.2)

    if result["winner"]:
        qs, summary, name = result["winner"]
        return qs, summary

    err = " | ".join(result["errors"]) or "Timeout"
    raise RuntimeError(f"Race failed — {err}")


# ══════════════════════════════════════════════════════════════
# GEMINI BACKEND
# ══════════════════════════════════════════════════════════════

def _gemini_call(api_key, model, prompt, retries=2):
    client = genai.Client(api_key=api_key)
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=8192,
                ),
            )
            return resp.text
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ["429", "quota", "RESOURCE_EXHAUSTED"]) and attempt < retries - 1:
                wait = 10 + random.uniform(0, 3)
                print(f"    ⏳ Gemini waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                raise


# ══════════════════════════════════════════════════════════════
# GROQ BACKEND
# ══════════════════════════════════════════════════════════════

def _get_groq_models(api_key):
    try:
        import requests as _req
        r = _req.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json().get("data", [])
            skip = {"whisper", "guard", "tts", "safeguard", "vision"}
            ids  = [
                m["id"] for m in data
                if m.get("active", True)
                and not any(s in m["id"].lower() for s in skip)
            ]
            big   = [i for i in ids if any(x in i for x in ["70b", "120b", "32b", "scout", "maverick"])]
            small = [i for i in ids if i not in big]
            ordered = big + small
            if ordered:
                print(f"    🔍 Groq live models: {ordered[:3]}")
                return ordered
    except Exception as e:
        print(f"    ⚠️ Could not fetch Groq model list: {e}")
    return GROQ_MODELS


def _groq_call(api_key, model, prompt, retries=2):
    client  = Groq(api_key=api_key)
    trimmed = prompt if len(prompt) <= 20000 else prompt[:20000] + "\n\n[Content trimmed. Generate questions from the above only.]"
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an educational question generator. "
                            "STRICT OUTPUT RULES:\n"
                            "1. Respond with ONLY a raw JSON array — nothing else.\n"
                            "2. Start your response with [ and end with ].\n"
                            "3. No markdown, no code fences, no prose, no explanation.\n"
                            "4. ALL string values must be plain readable text — "
                            "NEVER put HTML, XML, or any angle-bracket tags inside JSON string values.\n"
                            "5. question_en must be a real study question in plain English."
                        ),
                    },
                    {"role": "user", "content": trimmed},
                ],
                temperature=0.7,
                max_tokens=6000,
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ["429", "rate_limit", "rate limit"]) and attempt < retries - 1:
                wait = 8 + random.uniform(0, 2)
                print(f"    ⏳ Groq waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                raise


# ══════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════

def _build_batch_prompt(
    content, filename, difficulties, lengths, qtype_instr,
    lang, meta_ctx, batch_size, id_start, page_from, page_to,
    wants_summary, already_asked, for_groq=False,
):
    bn_rule = (
        "Provide accurate Bangla for question_bn and answer_bn."
        if lang != "english"
        else 'Set question_bn and answer_bn to "".'
    )
    en_rule  = 'Set answer_en to "".' if lang == "bangla" else ""
    diff_str = ", ".join(d.capitalize() for d in difficulties)
    skip_block = ""
    if already_asked:
        shown = already_asked[:10]
        skip_block = "Do NOT repeat these questions:\n" + "\n".join(f"  - {q}" for q in shown) + "\n\n"

    summary_block = (
        "Write a short file summary in <SUMMARY>...</SUMMARY> tags BEFORE the JSON array.\n\n"
        if wants_summary else ""
    )

    # Groq models get a simpler schema — no HTML examples that confuse them
    schema = """[
  {{
    "id": {id_start},
    "question_en": "Write the question text here in plain English",
    "question_bn": "Bengali translation here or empty string",
    "difficulty": "easy",
    "length": "short",
    "important": false,
    "answer_en": "Write the answer here in plain English",
    "answer_bn": "Bengali translation of answer here or empty string",
    "formula": null,
    "topic": "Topic name",
    "section_name": "Section heading from the content",
    "search_keyword": "one keyword",
    "where_to_find": "Page X, under Heading Y",
    "question_type": "definition",
    "page_references": ["Page {page_from}"]
  }}
]""".format(id_start=id_start, page_from=page_from)

    return f"""You are generating study questions for a student named Onna.

File: "{filename}" | Pages: {page_from}–{page_to}
{meta_ctx}

CONTENT:
{content}

{summary_block}{skip_block}Generate exactly {batch_size} questions from the CONTENT above.
IDs must go from {id_start} to {id_start + batch_size - 1}.
Question focus: {qtype_instr}
Difficulty levels to include: {diff_str}
Answer length: {", ".join(lengths)} (short=2-3 sentences, long=detailed paragraph)
{bn_rule}
{en_rule}

CRITICAL RULES:
1. Output ONLY a valid JSON array — nothing before or after it.
2. Start with [ and end with ]. No markdown. No prose. No explanation.
3. question_en and answer_en must contain PLAIN TEXT only — no HTML, no tags, no angle brackets.
4. Every string value must be plain readable text.

Follow this exact JSON structure:
{schema}"""


# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════

def _build_meta_context(metadata):
    if not metadata:
        return ""
    parts = []
    if metadata.get("formulas"):
        parts.append("Key formulas: " + " | ".join(metadata["formulas"][:8]))
    if metadata.get("sections"):
        parts.append("Sections: " + " | ".join(metadata["sections"][:6]))
    return "\n".join(parts)


def _extract_summary(raw):
    m = re.search(r"<SUMMARY>(.*?)</SUMMARY>", raw, re.DOTALL)
    if m:
        return m.group(1).strip(), raw.replace(m.group(0), "").strip()
    return None, raw


def _parse_json(raw):
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # Direct parse
    try:
        r = json.loads(cleaned)
        if isinstance(r, list):
            return r
        if isinstance(r, dict):
            for v in r.values():
                if isinstance(v, list) and v:
                    return v
    except json.JSONDecodeError:
        pass

    # Find array by brackets
    s = cleaned.find("[")
    e = cleaned.rfind("]") + 1
    if s != -1 and e > s:
        try:
            r = json.loads(cleaned[s:e])
            if isinstance(r, list):
                return r
        except json.JSONDecodeError:
            pass

    # Object-by-object recovery for truncated output
    print("  ⚠️ Full parse failed — recovering objects individually")
    if s == -1:
        raise ValueError(f"No JSON array found in response:\n{raw[:300]}")
    partial   = cleaned[s:]
    recovered = []
    depth, obj_start = 0, None
    i = 0
    while i < len(partial):
        ch = partial[i]
        if ch == '"':
            i += 1
            while i < len(partial):
                if partial[i] == '\\':
                    i += 2
                    continue
                if partial[i] == '"':
                    break
                i += 1
        elif ch == '{':
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    obj = json.loads(partial[obj_start:i + 1])
                    if isinstance(obj, dict) and "question_en" in obj:
                        recovered.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None
        i += 1

    if recovered:
        print(f"  ✅ Recovered {len(recovered)} questions")
        return recovered
    raise ValueError(f"Could not parse any JSON from response:\n{raw[:400]}")


def _sanitize_questions(questions):
    """
    Clean up questions from Groq/Gemini:
    - Strip HTML tags from all text fields
    - Reject questions where question_en looks like HTML (bad generation)
    - Ensure question_en is a real question, not metadata/code
    """
    html_tag     = re.compile(r"<[^>]+>")
    html_pattern = re.compile(r"<(div|span|p|br|html|body|style|script)[\s>]", re.IGNORECASE)
    text_fields  = ["question_en", "question_bn", "answer_en", "answer_bn",
                    "topic", "section_name", "search_keyword", "where_to_find"]
    clean = []
    for q in questions:
        q_en = q.get("question_en", "")

        # Reject if question_en is clearly HTML garbage
        if html_pattern.search(q_en):
            print(f"  🗑️ Rejected bad question (HTML in question_en): {q_en[:60]}")
            continue

        # Reject if question_en is empty or too short after stripping
        q_en_stripped = html_tag.sub("", q_en).strip()
        if len(q_en_stripped) < 10:
            print(f"  🗑️ Rejected empty/short question_en: {repr(q_en[:60])}")
            continue

        # Strip HTML from all text fields
        for field in text_fields:
            val = q.get(field)
            if isinstance(val, str) and "<" in val:
                q[field] = html_tag.sub("", val).strip()

        clean.append(q)

    if len(clean) < len(questions):
        print(f"  ⚠️ Sanitizer removed {len(questions)-len(clean)} malformed questions")
    return clean
