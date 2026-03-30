"""
qa_generator.py
Each question now returns extra "find it" fields:
  - section_name   : e.g. "Chapter 3: Gibbs Energy"
  - search_keyword : a short keyword Onna can Ctrl+F in her file
  - where_to_find  : human-readable hint e.g. "Page 12, under Entropy"
"""

import json
import re
import time
import random
import concurrent.futures
from google import genai
from google.genai import types


AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
]

MAX_CHARS_PER_CHUNK = 30000
MAX_PARALLEL_CHUNKS = 3

QTYPE_INSTRUCTIONS = {
    "definition":  "Generate ONLY definition-type questions: 'What is...?', 'Define...', 'What do you mean by...?'. Focus on terms and their meanings.",
    "descriptive": "Generate ONLY descriptive questions: 'Explain...', 'Describe...', 'Discuss...', 'How does...?'. Answers must be detailed paragraphs.",
    "formula":     "Generate ONLY questions involving mathematical formulas, chemical equations, bonds, or numerical expressions. Every answer must include a formula.",
    "figure":      "Generate ONLY questions about diagrams, figures, graphs, or structural content: 'Draw and explain...', 'What does the diagram show?', 'Sketch and label...'",
    "smart":       "Pick the most important questions covering key concepts, frequently tested ideas, and foundational principles.",
    "mixed":       "Generate a balanced mix of definition, descriptive, and formula-based questions.",
}


def generate_questions(
    text, filename, api_key,
    difficulties, lengths,
    q_per_page, page_from, page_to, lang,
    metadata=None,
    smart_mode=False, smart_total=50,
    question_type="mixed", wants_summary=False,
):
    meta_ctx       = _build_meta_context(metadata)
    pages_in_range = max(1, page_to - page_from + 1)
    qtype_instr    = QTYPE_INSTRUCTIONS.get(question_type, QTYPE_INSTRUCTIONS["mixed"])

    # ── Smart mode ───────────────────────────────────────────────
    if smart_mode:
        prompt = _build_smart_prompt(
            content=text[:MAX_CHARS_PER_CHUNK * 2],
            filename=filename, difficulties=difficulties, lengths=lengths,
            total_q=smart_total, lang=lang, meta_ctx=meta_ctx,
            qtype_instr=qtype_instr, wants_summary=wants_summary,
        )
        questions = _call_api(api_key, prompt, filename, 0, 1)
        for q in questions:
            q["source_file"] = filename
        return questions

    # ── Single chunk ─────────────────────────────────────────────
    if len(text) <= MAX_CHARS_PER_CHUNK:
        target_q = max(len(difficulties),
                       min(pages_in_range * q_per_page * len(difficulties), 80))
        prompt = _build_prompt(
            content=text, filename=filename,
            difficulties=difficulties, lengths=lengths,
            target_q=target_q, q_per_page=q_per_page,
            page_start=page_from, page_end=page_to,
            lang=lang, meta_ctx=meta_ctx,
            qtype_instr=qtype_instr,
            wants_summary=wants_summary,
        )
        questions = _call_api(api_key, prompt, filename, 0, 1)
        for q in questions: q["source_file"] = filename
        return questions

    # ── Multiple chunks (parallel) ───────────────────────────────
    chunks = _split_text(text)
    print(f"  📄 {filename}: {len(chunks)} chunks")

    chunk_args = []
    for ci, chunk in enumerate(chunks):
        ppc  = max(1, pages_in_range // len(chunks))
        ps   = page_from + ci * ppc
        pe   = min(page_to, ps + ppc - 1)
        tq   = max(len(difficulties), min(ppc * q_per_page * len(difficulties), 60))
        prompt = _build_prompt(
            content=chunk, filename=filename,
            difficulties=difficulties, lengths=lengths,
            target_q=tq, q_per_page=q_per_page,
            page_start=ps, page_end=pe,
            lang=lang, meta_ctx=meta_ctx,
            qtype_instr=qtype_instr,
            wants_summary=(wants_summary and ci == 0),
        )
        chunk_args.append((api_key, prompt, filename, ci, len(chunks), ps, pe))

    all_questions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHUNKS) as ex:
        futures = {ex.submit(_call_api, a[0],a[1],a[2],a[3],a[4]): a for a in chunk_args}
        for fut in concurrent.futures.as_completed(futures):
            a = futures[fut]
            ps, pe = a[5], a[6]
            try:
                qs = fut.result()
                for q in qs:
                    q["source_file"] = filename
                    if len(chunks) > 1:
                        q.setdefault("topic","")
                        tag = f" [p.{ps}–{pe}]"
                        if tag not in q["topic"]:
                            q["topic"] = (q["topic"]+tag).strip()
                all_questions.extend(qs)
            except Exception as e:
                print(f"  ⚠️ Chunk {a[3]+1} failed: {e}")

    return all_questions


# ── Prompts ───────────────────────────────────────────────────────

def _build_smart_prompt(content, filename, difficulties, lengths, total_q,
                        lang, meta_ctx, qtype_instr, wants_summary):
    bn_rule = ("Provide accurate Bangla translations for question_bn and answer_bn."
               if lang != "english" else 'Set question_bn and answer_bn to "".')
    en_rule = 'Set answer_en to "".' if lang == "bangla" else ""
    per_diff = max(1, total_q // len(difficulties))
    diff_lines = "\n".join(f"  • {d.capitalize()}: {per_diff} questions" for d in difficulties)

    summary_block = ""
    if wants_summary:
        summary_block = 'ALSO: Write a SHORT SUMMARY in <SUMMARY>...</SUMMARY> tags before the JSON.\n'

    return f"""You are an expert bilingual educational content creator helping a student named Onna.

Source: "{filename}"
{meta_ctx}

CONTENT:
{content}

{summary_block}

TASK: Pick and generate the {total_q} MOST IMPORTANT questions from this content.
Question focus: {qtype_instr}

Distribution across difficulties:
{diff_lines}
Lengths: distribute evenly between {", ".join(lengths)}

RULES:
- Mark ALL as "important": true
- {bn_rule}
- {en_rule}
- short = 2–3 sentence answer, long = detailed multi-paragraph
- Include formula/equation in formula field when relevant
- For EACH question, fill in these "find it" fields so Onna can locate it in her file:
    • section_name    : the chapter/section heading this topic belongs to (e.g. "3.2 Entropy")
    • search_keyword  : ONE short word or phrase Onna can Ctrl+F to find this in her file
    • where_to_find   : plain English hint, e.g. "Page 12, under the heading Gibbs Free Energy"
    • page_references : list of page numbers, e.g. ["Page 12", "Page 13"]

Return a COMPLETE valid JSON array (after summary tags if any).
No markdown. No truncation. Complete every object fully.

[{{
  "id": 1,
  "question_en": "...",
  "question_bn": "...",
  "difficulty": "beginner|easy|intermediate|hard",
  "length": "short|long",
  "important": true,
  "answer_en": "...",
  "answer_bn": "...",
  "formula": "...or null",
  "topic": "topic name",
  "section_name": "e.g. Chapter 3: Thermodynamics",
  "search_keyword": "e.g. entropy",
  "where_to_find": "e.g. Page 5, under heading Entropy",
  "question_type": "definition|descriptive|formula|figure|mixed",
  "page_references": ["Page 1"]
}}]"""


def _build_prompt(content, filename, difficulties, lengths, target_q, q_per_page,
                  page_start, page_end, lang, meta_ctx, qtype_instr, wants_summary):
    bn_rule = ("Provide accurate Bangla translations for question_bn and answer_bn."
               if lang != "english" else 'Set question_bn and answer_bn to "".')
    en_rule = 'Set answer_en to "".' if lang == "bangla" else ""
    diff_lines = "\n".join(
        f"  • {d.capitalize()}: {q_per_page} q/page × {max(1,page_end-page_start+1)} pages"
        for d in difficulties)
    summary_block = 'ALSO: Write a SHORT SUMMARY in <SUMMARY>...</SUMMARY> tags before the JSON.\n' \
                    if wants_summary else ""

    return f"""You are an expert bilingual educational content creator helping Onna study.

Source: "{filename}" | Pages: {page_start}–{page_end}
{meta_ctx}

CONTENT:
{content}

{summary_block}

TASK: Generate exactly {target_q} questions:
{diff_lines}
Question focus: {qtype_instr}
Lengths: {", ".join(lengths)} (short=2-3 sentences, long=multi-paragraph)

RULES:
- Questions ONLY from this content
- Mark ~20% as "important": true
- Include formula/law in formula field when present
- {bn_rule}
- {en_rule}
- For EACH question fill in "find it" fields:
    • section_name   : heading/chapter this belongs to (e.g. "2.1 Ideal Gas Law")
    • search_keyword : one short Ctrl+F keyword from the file
    • where_to_find  : e.g. "Page 8, under Ideal Gas"
    • page_references: e.g. ["Page 8"]

Return COMPLETE valid JSON array. No markdown. No truncation.

[{{
  "id": 1,
  "question_en": "...",
  "question_bn": "...",
  "difficulty": "beginner|easy|intermediate|hard",
  "length": "short|long",
  "important": false,
  "answer_en": "...",
  "answer_bn": "...",
  "formula": "...or null",
  "topic": "...",
  "section_name": "...",
  "search_keyword": "...",
  "where_to_find": "...",
  "question_type": "definition|descriptive|formula|figure|mixed",
  "page_references": ["Page {page_start}"]
}}]"""


def _build_meta_context(metadata):
    if not metadata: return ""
    parts = []
    if metadata.get("formulas"):
        parts.append("KEY FORMULAS:\n"+"\n".join(f"  • {f}" for f in metadata["formulas"][:10]))
    if metadata.get("sections"):
        parts.append("SECTIONS:\n"+"\n".join(f"  • {s}" for s in metadata["sections"][:8]))
    return "\n".join(parts)


# ── API ───────────────────────────────────────────────────────────

def _call_api(api_key, prompt, filename, chunk_idx, total_chunks):
    client = genai.Client(api_key=api_key)
    failed = set()
    for model_name in AVAILABLE_MODELS:
        if model_name in failed: continue
        try:
            print(f"  → {model_name} | chunk {chunk_idx+1}/{total_chunks}")
            raw = _call_with_retry(client, model_name, prompt)

            summary = None
            m = re.search(r"<SUMMARY>(.*?)</SUMMARY>", raw, re.DOTALL)
            if m:
                summary = m.group(1).strip()
                raw = raw.replace(m.group(0),"").strip()

            questions = _parse_json(raw)
            for q in questions:
                q["model_used"] = model_name
                if summary:
                    q["_summary"] = summary
                    summary = None
            print(f"  ✅ {model_name} → {len(questions)} questions")
            return questions

        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ["429","quota","RESOURCE_EXHAUSTED","rate"]):
                print(f"  ⚠️ {model_name} rate limited → next"); failed.add(model_name); time.sleep(3)
            elif any(x in msg for x in ["404","not found","NOT_FOUND"]):
                print(f"  ⚠️ {model_name} not found → next"); failed.add(model_name)
            else:
                raise RuntimeError(f"Gemini error ({model_name}): {e}")
    raise RuntimeError("All models failed. Wait 1 minute and try again.")


def _call_with_retry(client, model_name, prompt, max_retries=2):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7, max_output_tokens=65535),
            )
            return response.text
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ["429","quota","RESOURCE_EXHAUSTED"]) and attempt < max_retries-1:
                wait = 10 + random.uniform(0,3)
                m = re.search(r"retry[_\s]after[:\s]*([\d.]+)", msg, re.IGNORECASE)
                if m: wait = float(m.group(1))+1
                print(f"  ⏳ Waiting {wait:.0f}s"); time.sleep(wait)
            else: raise


def _parse_json(raw):
    cleaned = re.sub(r"```(?:json)?","",raw).replace("```","").strip()
    try:
        r = json.loads(cleaned)
        if isinstance(r,list): return r
        if isinstance(r,dict):
            for v in r.values():
                if isinstance(v,list): return v
    except json.JSONDecodeError: pass

    s = cleaned.find("["); e = cleaned.rfind("]")+1
    if s != -1 and e > s:
        try:
            r = json.loads(cleaned[s:e])
            if isinstance(r,list): return r
        except json.JSONDecodeError: pass

    # Truncation recovery
    print("  ⚠️  Truncated — recovering...")
    if s == -1: raise ValueError(f"No JSON:\n{raw[:300]}")
    partial   = cleaned[s:]
    recovered = []
    depth, obj_start = 0, None
    i = 0
    while i < len(partial):
        ch = partial[i]
        if ch == '"':
            i += 1
            while i < len(partial):
                if partial[i]=='\\': i+=2; continue
                if partial[i]=='"': break
                i += 1
        elif ch=='{':
            if depth==0: obj_start=i
            depth+=1
        elif ch=='}':
            depth-=1
            if depth==0 and obj_start is not None:
                try:
                    obj=json.loads(partial[obj_start:i+1])
                    if isinstance(obj,dict) and "question_en" in obj:
                        recovered.append(obj)
                except json.JSONDecodeError: pass
                obj_start=None
        i+=1

    if recovered:
        print(f"  ✅ Recovered {len(recovered)} questions"); return recovered
    raise ValueError(f"Could not parse JSON:\n{raw[:400]}")


def _split_text(text, max_chars=MAX_CHARS_PER_CHUNK):
    if len(text)<=max_chars: return [text]
    chunks=[]
    while text:
        if len(text)<=max_chars: chunks.append(text); break
        cut=text.rfind("\n\n",0,max_chars)
        if cut==-1: cut=text.rfind("\n",0,max_chars)
        if cut==-1: cut=text.rfind(". ",0,max_chars)
        if cut==-1: cut=max_chars
        chunks.append(text[:cut+1]); text=text[cut+1:].strip()
    return chunks
