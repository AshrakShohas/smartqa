"""
qa_generator.py
Sends extracted text to Google Gemini and returns structured Q&A.
With auto-fallback model selection and rate limit handling.
"""

import json
import re
import time
import random
import google.generativeai as genai


# List of available models in order of preference
# If one hits rate limits, it will automatically try the next one
AVAILABLE_MODELS = [
    "gemini-flash-latest",      # Auto-updates to latest flash model
    "gemini-2.0-flash",         # Stable and fast
    "gemini-2.5-flash",         # Newer model
    "gemini-pro-latest",        # Older but very stable
    "gemini-1.5-flash",         # Legacy (may work)
]


def generate_questions(
    text: str,
    filename: str,
    api_key: str,
    difficulties: list,
    lengths: list,
    q_per_diff: int,
    lang: str,
    metadata: dict = None,  # New parameter for metadata
) -> list:
    """
    Calls Gemini API and returns a list of question dicts.
    Handles long texts by chunking.
    """
    genai.configure(api_key=api_key)
    
    # Prepare metadata info for prompt
    metadata_info = ""
    if metadata:
        if metadata.get("formulas"):
            metadata_info += f"\n\n📐 KEY FORMULAS FOUND:\n" + "\n".join(['- ' + f for f in metadata['formulas'][:10]])
        if metadata.get("sections"):
            metadata_info += f"\n\n📑 SECTION HEADERS:\n" + "\n".join(['- ' + s for s in metadata['sections'][:5]])
        if metadata.get("pages"):
            metadata_info += f"\n\n📄 DOCUMENT PAGES: {len(metadata['pages'])} pages total"
        if metadata.get("images"):
            metadata_info += f"\n\n🖼️ CONTAINS IMAGES: Yes"
    
    # Split text into chunks
    chunks = _split_text(text, max_chars=12000)
    all_questions = []
    
    # Keep track of failed models to avoid retrying them
    failed_models = set()
    
    for chunk_idx, chunk in enumerate(chunks):
        # Add page numbers to chunk if available
        if metadata and metadata.get("pages") and len(metadata["pages"]) > chunk_idx:
            chunk = f"[From page/section {metadata['pages'][min(chunk_idx, len(metadata['pages'])-1)]}]\n{chunk}"
        
        # Try to generate questions for this chunk
        questions = _generate_with_fallback(
            chunk=chunk,
            filename=filename,
            difficulties=difficulties,
            lengths=lengths,
            q_per_diff=q_per_diff,
            lang=lang,
            chunk_idx=chunk_idx,
            total_chunks=len(chunks),
            failed_models=failed_models,
            metadata_info=metadata_info  # Pass metadata info
        )
        all_questions.extend(questions)
        
        # Small delay between chunks to respect rate limits
        if len(chunks) > 1:
            time.sleep(2)
    
    return all_questions


def _generate_with_fallback(
    chunk: str,
    filename: str,
    difficulties: list,
    lengths: list,
    q_per_diff: int,
    lang: str,
    chunk_idx: int,
    total_chunks: int,
    failed_models: set,
    metadata_info: str = ""  # Add this parameter with default value
) -> list:
    """
    Attempts to generate questions using available models.
    Switches models on failure with exponential backoff.
    """
    # Create a list of models to try (excluding failed ones)
    models_to_try = [m for m in AVAILABLE_MODELS if m not in failed_models]
    
    if not models_to_try:
        raise RuntimeError("All models have failed. Please check your API key and try again later.")
    
    last_error = None
    
    for model_name in models_to_try:
        try:
            print(f"Attempting with model: {model_name} (chunk {chunk_idx+1}/{total_chunks})")
            
            # Create model instance
            model = genai.GenerativeModel(model_name)
            
            # Build prompt with metadata
            prompt = _build_prompt(
                chunk, difficulties, lengths, q_per_diff, lang, filename, metadata_info
            )
            
            # Try with exponential backoff for rate limits
            questions = _generate_with_retry(model, prompt, model_name, filename, chunk_idx, total_chunks)
            
            # Success! Return questions
            return questions
            
        except Exception as e:
            error_msg = str(e)
            last_error = e
            
            # Check if it's a quota/rate limit error
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"⚠️ Model {model_name} hit rate limit. Trying next model...")
                failed_models.add(model_name)
                
                # Wait before trying next model (respect rate limits)
                wait_time = 5
                if "retry in" in error_msg:
                    import re
                    match = re.search(r"retry in ([\d.]+)s", error_msg)
                    if match:
                        wait_time = float(match.group(1))
                
                print(f"⏳ Waiting {wait_time} seconds before trying next model...")
                time.sleep(min(wait_time, 30))  # Cap at 30 seconds
                continue
                
            elif "404" in error_msg:
                print(f"⚠️ Model {model_name} not found. Trying next model...")
                failed_models.add(model_name)
                continue
                
            else:
                # Other error - raise immediately
                raise RuntimeError(f"Gemini API error with {model_name}: {e}")
    
    # If we get here, all models failed
    raise RuntimeError(f"All models failed. Last error: {last_error}")


def _generate_with_retry(model, prompt: str, model_name: str, filename: str, chunk_idx: int, total_chunks: int) -> list:
    """
    Attempts to generate content with exponential backoff for rate limits.
    """
    max_retries = 3
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            raw = response.text
            questions = _parse_json(raw)
            
            # Tag source file and chunk info
            for q in questions:
                q["source_file"] = filename
                q["model_used"] = model_name  # Track which model generated this
                if total_chunks > 1:
                    q["topic"] = q.get("topic", "") + f" (chunk {chunk_idx+1}/{total_chunks})"
            
            print(f"✅ Success with {model_name} (attempt {attempt + 1})")
            return questions
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error
            if "429" in error_msg or "quota" in error_msg.lower():
                if attempt < max_retries - 1:
                    # Extract wait time if provided
                    wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                    if "retry in" in error_msg:
                        import re
                        match = re.search(r"retry in ([\d.]+)s", error_msg)
                        if match:
                            wait_time = float(match.group(1))
                    
                    # Add jitter to avoid thundering herd
                    wait_time += random.uniform(0, 2)
                    
                    print(f"⚠️ Rate limit on {model_name}. Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise  # Max retries exceeded
            
            # Not a rate limit error - raise immediately
            raise


def _split_text(text: str, max_chars: int = 12000) -> list:
    """Split text into chunks without breaking sentences."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_chars:
            chunks.append(text)
            break
        # Find last sentence end before max_chars
        cut = text.rfind(".", 0, max_chars)
        if cut == -1:
            cut = max_chars
        chunks.append(text[:cut + 1])
        text = text[cut + 1:].strip()

    return chunks


def _build_prompt(content, difficulties, lengths, q_per_diff, lang, filename, metadata_info=""):
    bangla_instruction = ""
    if lang != "english":
        bangla_instruction = "- Provide accurate Bangla translations for question_bn and answer_bn."
    else:
        bangla_instruction = '- Set question_bn and answer_bn to empty string "".'

    en_instruction = ""
    if lang == "bangla":
        en_instruction = '- Set answer_en to empty string "".'

    return f"""You are an expert bilingual educational content creator specializing in Bengali and English.

Analyze the following academic content from the file "{filename}" and generate questions based ONLY on it.

{metadata_info}

CONTENT:
{content}

INSTRUCTIONS:
- Generate exactly {q_per_diff} questions for EACH of these difficulty levels: {", ".join(difficulties)}
- Each question must use one of these lengths: {", ".join(lengths)} (distribute evenly)
- Mark approximately 20% of the most conceptually important questions with "important": true
- For "long" questions: write detailed, multi-paragraph answers covering all aspects
- For "short" questions: provide concise 2-3 sentence answers
- **IMPORTANT:** When answering, include SPECIFIC page numbers, slide numbers, or section references from the content
- Extract and highlight formulas FIRST before generating questions
- ALL questions must be strictly based on the provided content

LANGUAGE RULES:
{bangla_instruction}
{en_instruction}

OUTPUT FORMAT:
Return ONLY a valid JSON array. No markdown code fences. No explanation. No preamble.
Follow this exact schema for every item:

[
  {{
    "id": 1,
    "question_en": "Full English question here?",
    "question_bn": "বাংলায় প্রশ্ন এখানে?",
    "difficulty": "beginner",
    "length": "short",
    "important": false,
    "answer_en": "Full English answer here. Include page/section references.",
    "answer_bn": "বাংলায় সম্পূর্ণ উত্তর এখানে। Include page/section references.",
    "formula": "E = mc² (or null if not applicable)",
    "topic": "Topic name from content",
    "page_references": ["Page 5", "Section 2.1"]
  }}
]"""


def _parse_json(raw: str) -> list:
    """Robustly parse JSON from Gemini response."""
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # Try direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find JSON array in text
    start = cleaned.find("[")
    end   = cleaned.rfind("]") + 1
    if start != -1 and end > start:
        try:
            result = json.loads(cleaned[start:end])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse Gemini response as JSON.\nRaw response:\n{raw[:500]}")