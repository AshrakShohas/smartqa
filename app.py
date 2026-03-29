import streamlit as st
import os
from dotenv import load_dotenv
from extractor import extract_text_from_file
from qa_generator import generate_questions

load_dotenv()

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Onna's SmartQA Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@300;400;500;600&display=swap');

    /* Dark theme */
    .stApp { background-color: #0b0b12; color: #e8e8f5; }
    section[data-testid="stSidebar"] { background-color: #111118; border-right: 1px solid #2a2a45; }

    h1, h2, h3 { color: #e8e8f5 !important; }

    /* Cards */
    .q-card {
        background: #13131f;
        border: 1px solid #2a2a45;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 14px;
        border-left: 4px solid #7c6dfa;
    }
    .q-card.easy         { border-left-color: #6dfabd; }
    .q-card.beginner     { border-left-color: #a8fa6d; }
    .q-card.intermediate { border-left-color: #6db8fa; }
    .q-card.hard         { border-left-color: #fa6d8e; }
    .q-card.important    { border-left-color: #f5c518; }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 100px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-beginner     { background: rgba(168,250,109,0.12); color: #a8fa6d; border: 1px solid rgba(168,250,109,0.3); }
    .badge-easy         { background: rgba(109,250,189,0.12); color: #6dfabd; border: 1px solid rgba(109,250,189,0.3); }
    .badge-intermediate { background: rgba(109,184,250,0.12); color: #6db8fa; border: 1px solid rgba(109,184,250,0.3); }
    .badge-hard         { background: rgba(250,109,142,0.12); color: #fa6d8e; border: 1px solid rgba(250,109,142,0.3); }
    .badge-important    { background: rgba(245,197,24,0.12);  color: #f5c518; border: 1px solid rgba(245,197,24,0.3);  }
    .badge-short        { background: rgba(124,109,250,0.12); color: #7c6dfa; border: 1px solid rgba(124,109,250,0.3); }
    .badge-long         { background: rgba(250,109,142,0.12); color: #fa6d8e; border: 1px solid rgba(250,109,142,0.3); }

    .q-english  { font-size: 15px; font-weight: 600; margin: 10px 0 6px; color: #e8e8f5; line-height: 1.6; }
    .q-bangla   { font-family: 'Hind Siliguri', sans-serif; font-size: 15px; color: #9898b8;
                  background: #1c1c2e; border-left: 3px solid #7c6dfa; padding: 8px 12px;
                  border-radius: 6px; margin: 6px 0; line-height: 1.9; }
    .ans-en     { font-size: 13px; color: #c8c8e5; line-height: 1.75; white-space: pre-wrap; margin: 6px 0; }
    .ans-bn     { font-family: 'Hind Siliguri', sans-serif; font-size: 15px; color: #9898b8;
                  background: #1c1c2e; border-left: 3px solid #6dfabd; padding: 10px 14px;
                  border-radius: 6px; line-height: 1.95; white-space: pre-wrap; margin: 6px 0; }
    .formula    { background: #22223a; border: 1px solid #2a2a45; border-radius: 8px;
                  padding: 10px 14px; font-family: monospace; font-size: 13px; color: #6dfabd;
                  margin: 8px 0; }
    .formula::before { content: "FORMULA / সূত্র: "; font-size: 9px; letter-spacing: 2px;
                       color: #6dfabd88; text-transform: uppercase; }

    .stButton button {
        background: linear-gradient(135deg, #7c6dfa, #fa6d8e) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 700 !important;
        font-size: 15px !important; padding: 10px 24px !important;
        width: 100% !important;
    }
    .stButton button:hover { opacity: 0.9 !important; }

    div[data-testid="stFileUploader"] {
        background: #13131f;
        border: 1.5px dashed #2a2a45;
        border-radius: 14px;
        padding: 10px;
    }

    .stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0; }
    .stat-chip { background: #13131f; border: 1px solid #2a2a45; border-radius: 100px;
                 padding: 4px 14px; font-size: 12px; }
    
    .instruction-box {
        background: #1c1c2e;
        border-left: 4px solid #7c6dfa;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    .instruction-box h3 {
        margin-top: 0;
        color: #7c6dfa !important;
    }
    
    .instruction-box p {
        margin-bottom: 0;
        color: #c8c8e5;
        line-height: 1.6;
    }
    
    .success-message {
        background: rgba(109,250,189,0.1);
        border: 1px solid rgba(109,250,189,0.3);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────
if "questions" not in st.session_state:
    st.session_state.questions = []
if "generating" not in st.session_state:
    st.session_state.generating = False
if "model_used" not in st.session_state:
    st.session_state.model_used = None


# ── Sidebar settings ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # API Key - pre-filled with your key
    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        value="AIzaSyCQqjEc1_nM_0AkOuBKFeOU33MfetV9IM8",
        help="Your API key is already saved. You don't need to change it unless it stops working."
    )
    st.caption("✅ API Key is set and ready to use!")
    st.divider()
    
    st.markdown("### 📝 Question Settings")
    
    # Instructions in sidebar
    st.markdown("""
    <div style='background:#1c1c2e; padding:10px; border-radius:8px; margin-bottom:15px;'>
        <small>💡 <strong>Tip:</strong> Start with 2 questions per difficulty and adjust if you need more!</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Difficulty Levels")
    diff_beginner     = st.checkbox("🌱 Beginner",     value=True)
    diff_easy         = st.checkbox("✅ Easy",         value=True)
    diff_intermediate = st.checkbox("🔷 Intermediate", value=True)
    diff_hard         = st.checkbox("🔥 Hard",         value=True)
    st.divider()
    
    st.markdown("#### Answer Length")
    len_short = st.checkbox("Short answers (2-3 sentences)", value=True)
    len_long  = st.checkbox("Long answers (detailed paragraphs)",  value=True)
    st.divider()
    
    q_per_diff = st.slider(
        "Questions per difficulty level", 
        1, 10, 2,  # Changed default to 2
        help="How many questions to create for each difficulty level you selected"
    )
    
    answer_lang = st.selectbox(
        "Answer Language", 
        ["English + Bangla", "English Only", "Bangla Only"],
        help="Choose whether you want answers in English, Bangla, or both"
    )
    st.divider()
    
    st.markdown("### 📄 File Settings")
    chunk_size = st.slider(
        "Pages per chunk",
        min_value=5, max_value=50, value=10,  # Changed default to 10
        help="For large files: Smaller chunks = more API calls but better for big files. Default 10 pages works well."
    )
    
    st.divider()
    st.markdown("### ℹ️ About")
    st.caption("""
    This tool helps you create study questions from your files.
    Upload PDF, Word, or PowerPoint files and get questions with answers in English and Bangla!
    """)


# ── Main area ────────────────────────────────────────────────────
# Header with wife's name
st.markdown("""
<div style='text-align:center; padding: 10px 0 24px'>
  <div style='display:inline-block; background:linear-gradient(135deg,#7c6dfa,#fa6d8e);
              padding:4px 16px; border-radius:100px; font-size:11px; letter-spacing:3px;
              text-transform:uppercase; margin-bottom:14px; font-weight:600'>
    ✨ Made with Love for Onna ✨
  </div>
  <h1 style='font-size:2.4rem; font-weight:900; margin:0;
             background:linear-gradient(135deg,#e8e8f5,#7c6dfa,#fa6d8e);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent'>
    Onna's SmartQA Generator 📚
  </h1>
  <p style='color:#8888aa; margin-top:8px'>
    Upload your study materials → Get instant questions with answers in English & বাংলা
  </p>
</div>
""", unsafe_allow_html=True)

# Instructions Box - Clear and easy to understand
st.markdown("""
<div class="instruction-box">
    <h3>📖 How to Use (Easy Steps)</h3>
    <p>
    <strong>1️⃣ Upload files</strong> - Click below and select your PDF, Word, or PowerPoint files<br>
    <strong>2️⃣ Check settings</strong> - Adjust difficulty and language if needed (everything is already set up)<br>
    <strong>3️⃣ Click generate</strong> - Press the big button and wait a moment<br>
    <strong>4️⃣ Study!</strong> - Questions will appear below. You can export them to save or print<br>
    </p>
    <p style='margin-top:10px; font-size:13px;'>
    ⚡ <strong>Tip:</strong> Start with 1-2 files to see how it works, then add more!
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "📂 Upload your study files (PDF, DOCX, PPTX)",
    type=["pdf", "docx", "pptx", "ppt"],
    accept_multiple_files=True,
    help="You can upload one file or multiple files at once"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) ready! " + ", ".join([f.name for f in uploaded_files[:3]]) +
               (f" ... and {len(uploaded_files)-3} more" if len(uploaded_files) > 3 else ""))

# Center the generate button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_clicked = st.button("✨ Generate Questions Now!", use_container_width=True)


# ── Generate logic ───────────────────────────────────────────────
if generate_clicked:
    # Validation with clear messages
    if not api_key:
        st.error("❌ API Key is missing! Please contact support.")
        st.stop()
    if not uploaded_files:
        st.warning("📂 Please upload at least one file first (PDF, DOCX, or PPTX)")
        st.stop()

    selected_diffs = []
    if diff_beginner:     selected_diffs.append("beginner")
    if diff_easy:         selected_diffs.append("easy")
    if diff_intermediate: selected_diffs.append("intermediate")
    if diff_hard:         selected_diffs.append("hard")

    selected_lens = []
    if len_short: selected_lens.append("short")
    if len_long:  selected_lens.append("long")

    if not selected_diffs:
        st.warning("⚠️ Please select at least one difficulty level (Beginner, Easy, Intermediate, or Hard)")
        st.stop()
    if not selected_lens:
        st.warning("⚠️ Please select at least one answer length (Short or Long)")
        st.stop()

    lang_map = {"English + Bangla": "both", "English Only": "english", "Bangla Only": "bangla"}
    lang = lang_map[answer_lang]

    all_questions = []
    progress_bar  = st.progress(0)
    status_text   = st.empty()
    
    # Show which models are being tried
    model_status = st.empty()

    total = len(uploaded_files)

    # In the generate loop, update to handle metadata
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.markdown(f"📄 Reading **{uploaded_file.name}** ({idx+1}/{total})…")
        progress_bar.progress(int((idx / total) * 50))

        try:
            # Extract text AND metadata
            text, metadata = extract_text_from_file(uploaded_file, chunk_size=chunk_size)
            
            if not text.strip():
                st.warning(f"⚠️ Could not read text from {uploaded_file.name}. The file might be empty or protected.")
                continue
            
            # Show formulas if found
            if metadata.get("formulas"):
                with st.expander(f"📐 Formulas found in {uploaded_file.name}"):
                    for formula in metadata["formulas"][:10]:
                        st.code(formula, language=None)
            
            # Show image preview if it's an image
            if metadata.get("image_data"):
                st.image(metadata["image_data"], caption=uploaded_file.name, use_container_width=True)
            
            status_text.markdown(f"🤖 Creating questions from **{uploaded_file.name}**...")
            model_status.info("🔄 Trying different AI models if needed... (this is automatic)")

            qs = generate_questions(
                text=text,
                filename=uploaded_file.name,
                api_key=api_key,
                difficulties=selected_diffs,
                lengths=selected_lens,
                q_per_diff=q_per_diff,
                lang=lang,
                metadata=metadata  # Pass metadata
            )
            
            # Track which model was used
            if qs and qs[0].get("model_used"):
                st.session_state.model_used = qs[0]["model_used"]
                model_status.success(f"✅ Using model: {st.session_state.model_used}")
            
            all_questions.extend(qs)

        except Exception as e:
            st.warning(f"⚠️ Could not process {uploaded_file.name}: {e}")
            model_status.warning("🔄 Trying different model...")

        progress_bar.progress(int(((idx + 1) / total) * 100))

    progress_bar.progress(100)
    status_text.markdown(f"✅ Done! Created **{len(all_questions)} questions** from {total} file(s).")
    st.session_state.questions = all_questions


# ── Display questions ────────────────────────────────────────────
if st.session_state.questions:
    qs = st.session_state.questions
    total_q = len(qs)
    
    # Show which model was used
    if st.session_state.model_used:
        st.info(f"🤖 Questions generated using: **{st.session_state.model_used}**")

    counts = {
        "beginner":     sum(1 for q in qs if q.get("difficulty") == "beginner"),
        "easy":         sum(1 for q in qs if q.get("difficulty") == "easy"),
        "intermediate": sum(1 for q in qs if q.get("difficulty") == "intermediate"),
        "hard":         sum(1 for q in qs if q.get("difficulty") == "hard"),
        "important":    sum(1 for q in qs if q.get("important")),
        "short":        sum(1 for q in qs if q.get("length") == "short"),
        "long":         sum(1 for q in qs if q.get("length") == "long"),
    }

    st.markdown("---")
    st.markdown(f"### 📚 {total_q} Questions Ready for You!")

    # Stats
    stat_html = '<div class="stat-row">'
    stat_html += f'<span class="stat-chip">Total: {total_q}</span>'
    for k, label in [("beginner","🌱 Beginner"),("easy","✅ Easy"),("intermediate","🔷 Intermediate"),("hard","🔥 Hard"),("important","⭐ Important")]:
        if counts[k]:
            stat_html += f'<span class="stat-chip">{label}: {counts[k]}</span>'
    stat_html += '</div>'
    st.markdown(stat_html, unsafe_allow_html=True)
    
    # Filter options
    st.markdown("#### 🔍 Filter Questions")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_diff = st.multiselect("Filter by difficulty", ["beginner","easy","intermediate","hard"], default=["beginner","easy","intermediate","hard"])
    with col_f2:
        filter_len = st.multiselect("Filter by length", ["short","long"], default=["short","long"])
    with col_f3:
        show_important_only = st.checkbox("⭐ Show only important questions")

    # Apply filters
    filtered = [
        q for q in qs
        if q.get("difficulty","easy") in filter_diff
        and q.get("length","short") in filter_len
        and (not show_important_only or q.get("important"))
    ]

    st.caption(f"Showing {len(filtered)} of {total_q} questions")
    st.divider()

    # Export buttons
    st.markdown("#### 💾 Save Your Questions")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        import json
        st.download_button(
            "⬇️ Download as JSON (for computers)",
            data=json.dumps(qs, ensure_ascii=False, indent=2),
            file_name="onna_questions.json",
            mime="application/json",
            help="Best for saving and importing later"
        )
    with col_e2:
        # Build plain text export
        txt_lines = []
        for i, q in enumerate(filtered, 1):
            txt_lines.append(f"Q{i}. [{q.get('difficulty','').upper()}] [{q.get('length','').upper()}]{'  ⭐ IMPORTANT' if q.get('important') else ''}")
            txt_lines.append(f"EN: {q.get('question_en','')}")
            if q.get('question_bn'):
                txt_lines.append(f"BN: {q.get('question_bn','')}")
            txt_lines.append(f"\nAnswer (EN): {q.get('answer_en','')}")
            if q.get('answer_bn'):
                txt_lines.append(f"Answer (BN): {q.get('answer_bn','')}")
            if q.get('formula') and q.get('formula') != 'null':
                txt_lines.append(f"Formula: {q.get('formula')}")
            txt_lines.append("\n" + "-"*60 + "\n")
        st.download_button(
            "⬇️ Download as Text (printable)",
            data="\n".join(txt_lines),
            file_name="onna_questions.txt",
            mime="text/plain",
            help="Easy to print or save as notes"
        )

    # Render cards
    for i, q in enumerate(filtered):
        diff  = q.get("difficulty", "easy").lower()
        length = q.get("length", "short")
        imp   = q.get("important", False)
        card_class = "important" if imp else diff

        badges = ""
        if imp:
            badges += '<span class="badge badge-important">⭐ Important</span>'
        badges += f'<span class="badge badge-{diff}">{diff}</span>'
        badges += f'<span class="badge badge-{length}">{length}</span>'
        if q.get("topic"):
            badges += f'<span style="font-size:11px;color:#8888aa">{q["topic"]}</span>'

        formula_html = ""
        if q.get("formula") and q.get("formula") not in ("null", ""):
            formula_html = f'<div class="formula">{q["formula"]}</div>'

        ans_en_html = ""
        if q.get("answer_en"):
            ans_en_html = f'<div style="font-size:10px;letter-spacing:2px;color:#8888aa;text-transform:uppercase;margin-top:10px">📖 English Answer</div><div class="ans-en">{q["answer_en"]}</div>'

        ans_bn_html = ""
        if q.get("answer_bn"):
            ans_bn_html = f'<div style="font-size:10px;letter-spacing:2px;color:#8888aa;text-transform:uppercase;margin-top:10px">📖 বাংলা উত্তর</div><div class="ans-bn">{q["answer_bn"]}</div>'

        source_html = ""
        if q.get("source_file"):
            source_html = f'<div style="font-size:10px;color:#8888aa;margin-top:8px">📄 From: {q["source_file"]}</div>'

        card_html = f"""
        <div class="q-card {card_class}">
            <div>{badges}</div>
            <div class="q-english">Q{i+1}. {q.get("question_en","")}</div>
            {"<div class='q-bangla'>" + q.get("question_bn","") + "</div>" if q.get("question_bn") else ""}
            {source_html}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        with st.expander("📖 Show Answer"):
            st.markdown(f"""
            {ans_en_html}
            {formula_html}
            {ans_bn_html}
            """, unsafe_allow_html=True)
    
    # Clear button at bottom
    if st.button("🗑️ Clear All Questions", help="Remove all questions and start fresh"):
        st.session_state.questions = []
        st.session_state.model_used = None
        st.rerun()