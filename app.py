import streamlit as st
import os
import json
from dotenv import load_dotenv
from extractor import extract_text_from_file, get_page_count
from qa_generator import generate_questions

load_dotenv()
API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

st.set_page_config(
    page_title="Onna's SmartQA ✨",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@300;400;500;600&family=Nunito:wght@400;600;700;800;900&display=swap');

* { font-family: 'Nunito', sans-serif; }
.stApp { background-color: #0d0d1a; color: #e8e8f5; }
section[data-testid="stSidebar"] { background-color: #120d1f; border-right: 1px solid #2a1a45; }
h1,h2,h3 { color: #e8e8f5 !important; font-family: 'Nunito', sans-serif !important; }

.q-card {
    background: #13101f; border: 1px solid #2a1a45;
    border-radius: 16px; padding: 20px; margin-bottom: 14px;
    border-left: 4px solid #b56dfa;
}
.q-card.easy         { border-left-color: #6dfabd; }
.q-card.beginner     { border-left-color: #a8fa6d; }
.q-card.intermediate { border-left-color: #6db8fa; }
.q-card.hard         { border-left-color: #fa6d8e; }
.q-card.important    { border-left-color: #f5c518;
    background: linear-gradient(135deg,#1a150010,#13101f 60%); }

.badge { display:inline-block; padding:2px 10px; border-radius:100px; font-size:11px;
         font-weight:700; margin-right:5px; text-transform:uppercase; letter-spacing:1px; }
.badge-beginner     { background:rgba(168,250,109,0.12); color:#a8fa6d; border:1px solid rgba(168,250,109,0.3); }
.badge-easy         { background:rgba(109,250,189,0.12); color:#6dfabd; border:1px solid rgba(109,250,189,0.3); }
.badge-intermediate { background:rgba(109,184,250,0.12); color:#6db8fa; border:1px solid rgba(109,184,250,0.3); }
.badge-hard         { background:rgba(250,109,142,0.12); color:#fa6d8e; border:1px solid rgba(250,109,142,0.3); }
.badge-important    { background:rgba(245,197,24,0.15);  color:#f5c518; border:1px solid rgba(245,197,24,0.4); }
.badge-short        { background:rgba(181,109,250,0.12); color:#b56dfa; border:1px solid rgba(181,109,250,0.3); }
.badge-long         { background:rgba(250,109,142,0.12); color:#fa6d8e; border:1px solid rgba(250,109,142,0.3); }
.badge-type         { background:rgba(109,184,250,0.10); color:#6db8fa; border:1px solid rgba(109,184,250,0.25); }

.q-english { font-size:15px; font-weight:700; margin:10px 0 6px; color:#e8e8f5; line-height:1.6; }
.q-bangla  { font-family:'Hind Siliguri',sans-serif; font-size:15px; color:#b898d8;
             background:#1c1030; border-left:3px solid #b56dfa; padding:8px 12px;
             border-radius:6px; margin:6px 0; line-height:1.9; }
.ans-en    { font-size:13px; color:#c8c8e5; line-height:1.8; white-space:pre-wrap; margin:6px 0; }
.ans-bn    { font-family:'Hind Siliguri',sans-serif; font-size:15px; color:#b898d8;
             background:#1c1030; border-left:3px solid #6dfabd; padding:10px 14px;
             border-radius:6px; line-height:1.95; white-space:pre-wrap; margin:6px 0; }
.formula   { background:#1a1a2e; border:1px solid #2a2a55; border-radius:8px;
             padding:10px 14px; font-family:monospace; font-size:13px; color:#6dfabd; margin:8px 0; }
.formula::before { content:"📐 FORMULA / সূত্র:  "; font-size:9px; letter-spacing:2px;
                   color:#6dfabd88; text-transform:uppercase; }

/* ── Where to find it tag ── */
.find-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1a1035; border: 1px solid #3a2060;
    border-radius: 8px; padding: 5px 12px; margin: 8px 4px 4px 0;
    font-size: 12px; color: #c8a8f8;
}
.find-tag .icon { font-size: 14px; }
.find-tag .label { font-size: 9px; text-transform: uppercase;
                   letter-spacing: 1.5px; color: #8868a8; margin-right: 2px; }

.summary-box { background:linear-gradient(135deg,#1a0f2e,#0f1a2e); border:1px solid #3a2a55;
               border-radius:14px; padding:18px 22px; margin:10px 0; line-height:1.9;
               font-size:14px; color:#d8d8f0; }
.stButton button {
    background:linear-gradient(135deg,#9b4dfa,#fa4d8e) !important;
    color:white !important; border:none !important; border-radius:12px !important;
    font-weight:800 !important; font-size:16px !important; width:100% !important;
    padding:12px !important; box-shadow:0 4px 20px rgba(155,77,250,0.35) !important;
}
.stButton button:hover { opacity:0.92 !important; }
.page-chip { background:#1c1030; border:1px solid #3a2055; border-radius:10px;
             padding:9px 15px; margin:5px 0; font-size:13px; color:#b898d8; }
.info-box  { background:#160e28; border-left:4px solid #9b4dfa;
             padding:12px 16px; border-radius:10px; margin:10px 0;
             font-size:13px; color:#c8b8e8; line-height:1.7; }
.stat-row  { display:flex; gap:10px; flex-wrap:wrap; margin:12px 0; }
.stat-chip { background:#160e28; border:1px solid #3a2055; border-radius:100px;
             padding:4px 14px; font-size:12px; color:#c8b8e8; }

/* ── Page range card ── */
.range-card { background:#160e28; border:1px solid #3a2055; border-radius:14px;
              padding:16px 20px; margin:8px 0; }

/* ── Loading overlay ── */
.loading-overlay {
    position:fixed; inset:0; z-index:9999;
    background:rgba(13,13,26,0.97);
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    font-family:'Nunito',sans-serif;
}
.heart-pulse { font-size:80px; animation:heartbeat 1.2s ease-in-out infinite;
               filter:drop-shadow(0 0 30px rgba(250,100,140,0.7)); margin-bottom:20px; }
@keyframes heartbeat {
    0%,100%{transform:scale(1)} 14%{transform:scale(1.25)}
    28%{transform:scale(1)} 42%{transform:scale(1.2)} 70%{transform:scale(1)}
}
.sparkles { font-size:36px; letter-spacing:10px; margin-bottom:16px;
            animation:sparkle 2s ease-in-out infinite alternate; }
@keyframes sparkle {
    from{opacity:0.5;letter-spacing:10px} to{opacity:1;letter-spacing:18px}
}
.loading-title { font-size:2rem; font-weight:900; margin-bottom:10px;
    background:linear-gradient(135deg,#fa6db8,#b56dfa,#6db8fa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.loading-sub { font-size:15px; color:#b898d8; margin-bottom:30px; }
.loading-bar-wrap { width:320px; height:8px; background:#2a1a45;
                    border-radius:100px; overflow:hidden; margin-bottom:20px; }
.loading-bar { height:100%; border-radius:100px;
               background:linear-gradient(90deg,#9b4dfa,#fa4d8e,#6db8fa);
               background-size:200% 100%; animation:shimmer 1.5s ease-in-out infinite; }
@keyframes shimmer {
    0%{background-position:200% 0;width:20%}
    50%{background-position:0% 0;width:70%}
    100%{background-position:200% 0;width:90%}
}
.loading-msg { font-size:13px; color:#9878c8; font-style:italic; }
</style>
""", unsafe_allow_html=True)

for k,v in [("questions",[]),("summary",{}),("file_info",{}),("model_used",None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
      <div style='font-size:36px'>💜</div>
      <div style='font-weight:900;font-size:1.1rem;color:#b56dfa'>Onna's SmartQA</div>
      <div style='font-size:11px;color:#8868a8;margin-top:4px'>Made with love by Shohas 💕</div>
    </div>
    """, unsafe_allow_html=True)
    if API_KEY: st.success("🔑 Ready!")
    else:       st.error("🔑 API key missing in .env")
    st.divider()

    st.markdown("### 📊 Difficulty")
    diff_beginner     = st.checkbox("🌱 Beginner",     value=True)
    diff_easy         = st.checkbox("✅ Easy",         value=True)
    diff_intermediate = st.checkbox("🔷 Intermediate", value=True)
    diff_hard         = st.checkbox("🔥 Hard",         value=True)
    st.divider()

    st.markdown("### 📏 Answer Length")
    len_short = st.checkbox("Short  (2–3 sentences)",       value=True)
    len_long  = st.checkbox("Long   (detailed paragraphs)", value=True)
    st.divider()

    st.markdown("### 🌐 Language")
    answer_lang = st.selectbox("Answers in:",
        ["English + Bangla","English Only","Bangla Only"])
    st.divider()

    st.markdown("### ℹ️ Free API")
    st.caption("• 15 requests/min\n• 1,500/day\n• Auto model switch\n• No credit card")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:8px 0 24px'>
  <div style='font-size:48px;margin-bottom:8px'>📖💜</div>
  <h1 style='font-size:2.5rem;font-weight:900;margin:0;
    background:linear-gradient(135deg,#fa6db8,#b56dfa,#6db8fa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    Onna's SmartQA
  </h1>
  <p style='color:#8868a8;margin-top:8px;font-size:14px'>
    Your personal study companion 💕 — PDF · DOCX · PPTX → Smart Questions & Answers
  </p>
</div>
""", unsafe_allow_html=True)

if not API_KEY:
    st.error("❌ Add `GEMINI_API_KEY=your_key` to your `.env` file.")
    st.info("Free key at 👉 https://aistudio.google.com"); st.stop()


# ── STEP 1: Upload ───────────────────────────────────────────
st.markdown("## 📂 Step 1 — Upload Study Files")
st.caption("PDF, DOCX, PPTX, TXT — upload as many as you want!")

uploaded_files = st.file_uploader(
    "Drop files here", type=["pdf","docx","pptx","ppt","txt"],
    accept_multiple_files=True, label_visibility="collapsed"
)

file_info = {}
if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} file(s) ready:**")
    for f in uploaded_files:
        pc   = get_page_count(f); f.seek(0)
        ext  = f.name.rsplit(".",1)[-1].upper()
        unit = "slides" if ext in ("PPTX","PPT") else "pages"
        icon = {"PDF":"📄","DOCX":"📝","PPTX":"📊","PPT":"📊","TXT":"📃"}.get(ext,"📄")
        file_info[f.name] = {"page_count":pc,"file_type":ext,"unit":unit}
        st.markdown(
            f'<div class="page-chip">{icon} <strong>{f.name}</strong>'
            f' &nbsp;—&nbsp; {pc} {unit}</div>',
            unsafe_allow_html=True)
    st.session_state.file_info = file_info


# ── STEP 2 ────────────────────────────────────────────────────
if uploaded_files and st.session_state.file_info:
    st.markdown("---")
    st.markdown("## 🎯 Step 2 — What Do You Want to Study?")

    # ── Question type ─────────────────────────────────────────
    st.markdown("### 📌 Question Type")
    st.caption("Choose what kind of questions:")
    q_type = st.radio("Type", options=[
        "⭐ Smart Pick  — AI picks the most important questions automatically",
        "📖 Definitions  — What is...? / Define...",
        "📝 Descriptive  — Explain / Describe / Discuss",
        "🔢 Formulas & Equations  — Needs formula/equation/bond in answer",
        "🖼️ Figures & Diagrams  — Diagram/structure/graph related",
        "📋 Mixed  — All types combined",
    ], label_visibility="collapsed")

    wants_summary = st.checkbox(
        "📄 Also generate a short summary of each file", value=False)

    st.divider()
    st.markdown("### 🔢 How Many Questions?")
    smart_mode = "Smart Pick" in q_type

    # ── Mode A: Smart Pick ────────────────────────────────────
    if smart_mode:
        st.markdown("""
        <div class="info-box">
            ⭐ <strong>Smart Pick:</strong> AI reads your file and picks the most important
            questions. Set your target below — <strong>50 is perfect</strong> for a full file.
            Use <em>page range</em> if you want to focus on specific pages only.
        </div>
        """, unsafe_allow_html=True)

        per_file_smart = st.toggle("⚙️ Different settings per file", value=False)
        file_configs = {}

        if not per_file_smart:
            sc1, sc2 = st.columns(2)
            with sc1:
                smart_total_g = st.number_input(
                    "Total important questions (all files)",
                    min_value=5, max_value=200, value=50, step=5,
                    help="50 = ideal coverage without being overwhelming")
                n_d = sum([diff_beginner,diff_easy,diff_intermediate,diff_hard])
                st.info(f"⏱️ ~{max(1, smart_total_g // 25)} min estimated")
            with sc2:
                use_range_sg = st.toggle("📌 Specific page range only", value=False,
                                         key="smart_range_global")
                if use_range_sg and st.session_state.file_info:
                    max_p = max(v["page_count"] for v in st.session_state.file_info.values())
                    rsg   = st.slider("Page range", 1, max(max_p,2), (1,max_p),
                                      key="smart_slider_global",
                                      help="Only generate questions from these pages")
                else:
                    rsg = None

            for fname, info in st.session_state.file_info.items():
                pc = info["page_count"]
                pf, pt = rsg if rsg else (1, pc)
                file_configs[fname] = {
                    "smart_mode": True, "smart_total": smart_total_g,
                    "page_from": max(1,pf), "page_to": min(pc,pt), "q_per_page": 0,
                }

        else:
            for fname, info in st.session_state.file_info.items():
                pc, unit = info["page_count"], info["unit"]
                with st.expander(f"📄 {fname}  ({pc} {unit})", expanded=True):
                    ea, eb = st.columns(2)
                    with ea:
                        st_q = st.number_input(
                            "Important questions to pick",
                            min_value=5, max_value=200, value=50, step=5,
                            key=f"sq_{fname}")
                    with eb:
                        use_rp = st.toggle(f"📌 Specific {unit}", value=False, key=f"srt_{fname}")
                        if use_rp and pc > 1:
                            pr = st.slider(f"{unit.capitalize()} range", 1, pc, (1,pc),
                                           key=f"srng_{fname}")
                            pf, pt = pr
                            st.caption(f"→ From {unit} {pf} to {pt}")
                        else:
                            pf, pt = 1, pc
                    file_configs[fname] = {
                        "smart_mode": True, "smart_total": st_q,
                        "page_from": pf, "page_to": pt, "q_per_page": 0,
                    }

    # ── Mode B: Manual (per-page) ─────────────────────────────
    else:
        st.markdown("""
        <div class="info-box">
            💡 Set <em>questions per page</em> and optionally a <em>page range</em>.<br>
            Example: <strong>5 questions from pages 10–24</strong> of a specific chapter.
        </div>
        """, unsafe_allow_html=True)

        per_file_manual = st.toggle("⚙️ Different settings per file", value=False)
        file_configs = {}

        if not per_file_manual:
            mc1, mc2 = st.columns(2)
            with mc1:
                q_per_page_g = st.number_input(
                    "Questions per page (all files)",
                    min_value=1, max_value=10, value=2,
                    help="2 q/page × 37 pages = ~74 questions")
                total_pages = sum(v["page_count"] for v in st.session_state.file_info.values())
                n_d = sum([diff_beginner,diff_easy,diff_intermediate,diff_hard])
                est = total_pages * q_per_page_g * max(n_d,1)
                st.info(f"📊 Estimated: **~{est} questions** from {total_pages} pages")
            with mc2:
                use_range_g = st.toggle("📌 Specific page range only", value=False,
                                        key="manual_range_global")
                if use_range_g:
                    max_p = max(v["page_count"] for v in st.session_state.file_info.values())
                    rg    = st.slider("Page range (all files)", 1, max(max_p,2), (1,max_p),
                                      key="manual_slider_global")
                else:
                    rg = None

            for fname, info in st.session_state.file_info.items():
                pc = info["page_count"]
                pf,pt = (rg if rg else (1,pc))
                file_configs[fname] = {
                    "smart_mode": False, "q_per_page": q_per_page_g,
                    "page_from": max(1,pf), "page_to": min(pc,pt),
                }

        else:
            for fname, info in st.session_state.file_info.items():
                pc, unit = info["page_count"], info["unit"]
                with st.expander(f"📄 {fname}  ({pc} {unit})", expanded=True):
                    fa, fb = st.columns(2)
                    with fa:
                        qpp = st.number_input(
                            f"Questions per {unit[:-1]}",
                            min_value=1, max_value=10, value=2, key=f"qpp_{fname}")
                        n_d = sum([diff_beginner,diff_easy,diff_intermediate,diff_hard])
                        st.caption(f"→ ~{qpp * pc * max(n_d,1)} Q from all {unit}")
                    with fb:
                        use_r = st.toggle(f"📌 Specific {unit} only", value=False,
                                          key=f"rt_{fname}")
                        if use_r and pc > 1:
                            pr    = st.slider(f"{unit.capitalize()} range",
                                              1, pc, (1,pc), key=f"rng_{fname}")
                            pf,pt = pr
                            est_r = (pt-pf+1) * qpp * max(n_d,1)
                            st.caption(
                                f"→ {unit.capitalize()} **{pf} to {pt}** "
                                f"({pt-pf+1} {unit}) → ~{est_r} questions")
                        else:
                            pf,pt = 1,pc
                    file_configs[fname] = {
                        "smart_mode": False, "q_per_page": qpp,
                        "page_from": pf, "page_to": pt,
                    }


    # ── STEP 3: Generate ──────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🚀 Step 3 — Generate!")

    n_d = sum([diff_beginner,diff_easy,diff_intermediate,diff_hard])
    grand_total = 0
    for fname, cfg in file_configs.items():
        if cfg.get("smart_mode"):
            st.caption(f"• **{fname}**: Smart Pick → top {cfg['smart_total']} questions "
                       f"(pages {cfg['page_from']}–{cfg['page_to']})")
            grand_total += cfg["smart_total"]
        else:
            pages_used = cfg["page_to"] - cfg["page_from"] + 1
            est = pages_used * cfg["q_per_page"] * max(n_d,1)
            st.caption(f"• **{fname}**: pages {cfg['page_from']}–{cfg['page_to']} "
                       f"× {cfg['q_per_page']} q/page × {n_d} difficulty = ~{est} Q")
            grand_total += est
    st.success(f"🎯 Grand total: **~{grand_total} questions** will be generated")

    if st.button("✨ Generate Questions for Onna!"):
        if not any([diff_beginner,diff_easy,diff_intermediate,diff_hard]):
            st.error("❌ Select at least one difficulty."); st.stop()
        if not any([len_short,len_long]):
            st.error("❌ Select at least one length."); st.stop()

        selected_diffs = [d for d,on in [("beginner",diff_beginner),("easy",diff_easy),
                          ("intermediate",diff_intermediate),("hard",diff_hard)] if on]
        selected_lens  = [l for l,on in [("short",len_short),("long",len_long)] if on]
        lang = {"English + Bangla":"both","English Only":"english",
                "Bangla Only":"bangla"}[answer_lang]

        type_map = {
            "📖 Definitions":"definition","📝 Descriptive":"descriptive",
            "🔢 Formulas":"formula","🖼️ Figures":"figure",
            "📋 Mixed":"mixed","⭐ Smart":"smart",
        }
        qtype_key = next((v for k,v in type_map.items() if k[:3] in q_type), "mixed")

        # Loading animation
        loading_ph = st.empty()
        msgs = [
            "Reading your study files with love... 📖",
            "Finding the best questions for you... 🌟",
            "Almost ready, Onna! ✨",
            "Polishing your questions... 💜",
            "Just a little more... 💕",
        ]

        def show_loading(msg_idx=0, fname=""):
            loading_ph.markdown(f"""
            <div class="loading-overlay">
              <div class="sparkles">✨ 💜 ✨</div>
              <div class="heart-pulse">💖</div>
              <div class="loading-title">Crafting your questions!</div>
              <div class="loading-sub">Onna, this will just take a moment 💕</div>
              <div class="loading-bar-wrap"><div class="loading-bar"></div></div>
              <div class="loading-msg">{msgs[msg_idx % len(msgs)]}</div>
            </div>
            """, unsafe_allow_html=True)

        all_questions  = []
        all_summaries  = {}
        bar    = st.progress(0)
        status = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            fname = uploaded_file.name
            cfg   = file_configs.get(fname, {
                "smart_mode":False,"q_per_page":2,"page_from":1,"page_to":999})
            show_loading(idx, fname)
            bar.progress(int(idx / len(uploaded_files) * 40))
            status.markdown(f"📄 Reading **{fname}** "
                            f"(pages {cfg['page_from']}–{cfg['page_to']})...")

            try:
                text, metadata = extract_text_from_file(
                    uploaded_file,
                    page_from=cfg["page_from"],
                    page_to=cfg["page_to"],
                )
                if not text.strip():
                    st.warning(f"⚠️ No text in {fname} — skipping."); continue

                if metadata.get("formulas"):
                    with st.expander(f"📐 Formulas found in {fname}"):
                        for fm in metadata["formulas"][:15]:
                            st.code(fm, language=None)

                status.markdown(f"🤖 Generating for **{fname}**...")

                qs = generate_questions(
                    text=text, filename=fname, api_key=API_KEY,
                    difficulties=selected_diffs, lengths=selected_lens,
                    q_per_page=cfg.get("q_per_page",2),
                    page_from=cfg["page_from"], page_to=cfg["page_to"],
                    lang=lang, metadata=metadata,
                    smart_mode=cfg.get("smart_mode",False),
                    smart_total=cfg.get("smart_total",50),
                    question_type=qtype_key,
                    wants_summary=wants_summary,
                )

                # Pull summary out
                for q in qs:
                    if q.get("_summary") and fname not in all_summaries:
                        all_summaries[fname] = q.pop("_summary")

                if qs and qs[0].get("model_used"):
                    st.session_state.model_used = qs[0]["model_used"]

                all_questions.extend(qs)
                st.success(f"✅ **{fname}** → {len(qs)} questions")

            except Exception as e:
                st.warning(f"⚠️ {fname}: {e}")

            bar.progress(int((idx+1)/len(uploaded_files)*100))

        loading_ph.empty()
        bar.progress(100)
        status.markdown(f"🎉 Done! **{len(all_questions)} questions** ready for Onna!")
        st.session_state.questions = all_questions
        st.session_state.summary   = all_summaries
        st.balloons()


# ════════════════════════════════════════════════════════════
# DISPLAY
# ════════════════════════════════════════════════════════════
if st.session_state.questions:
    qs = st.session_state.questions

    if st.session_state.model_used:
        st.info(f"🤖 Generated using: **{st.session_state.model_used}**")

    if st.session_state.summary:
        st.markdown("---")
        st.markdown("## 📄 File Summaries")
        for fname, summ in st.session_state.summary.items():
            with st.expander(f"📋 Summary: {fname}"):
                st.markdown(f'<div class="summary-box">{summ}</div>',
                            unsafe_allow_html=True)

    counts = {
        "beginner":     sum(1 for q in qs if q.get("difficulty")=="beginner"),
        "easy":         sum(1 for q in qs if q.get("difficulty")=="easy"),
        "intermediate": sum(1 for q in qs if q.get("difficulty")=="intermediate"),
        "hard":         sum(1 for q in qs if q.get("difficulty")=="hard"),
        "important":    sum(1 for q in qs if q.get("important")),
        "short":        sum(1 for q in qs if q.get("length")=="short"),
        "long":         sum(1 for q in qs if q.get("length")=="long"),
    }

    st.markdown("---")
    st.markdown(f"### 💜 {len(qs)} Questions Ready for Onna!")

    stat_html = '<div class="stat-row">' + f'<span class="stat-chip">Total: {len(qs)}</span>'
    for k,label in [("beginner","🌱"),("easy","✅"),
                    ("intermediate","🔷"),("hard","🔥"),("important","⭐")]:
        if counts[k]:
            stat_html += f'<span class="stat-chip">{label} {k.capitalize()}: {counts[k]}</span>'
    st.markdown(stat_html+"</div>", unsafe_allow_html=True)

    f1,f2,f3 = st.columns(3)
    with f1:
        fd = st.multiselect("Difficulty",["beginner","easy","intermediate","hard"],
                            default=["beginner","easy","intermediate","hard"])
    with f2:
        fl = st.multiselect("Length",["short","long"],default=["short","long"])
    with f3:
        fi = st.checkbox("⭐ Important only")

    filtered = [q for q in qs
                if q.get("difficulty","easy") in fd
                and q.get("length","short") in fl
                and (not fi or q.get("important"))]
    st.caption(f"Showing {len(filtered)} of {len(qs)} questions")
    st.divider()

    ce1,ce2 = st.columns(2)
    with ce1:
        st.download_button("⬇️ Save as JSON",
            data=json.dumps(qs,ensure_ascii=False,indent=2),
            file_name="onna_questions.json", mime="application/json")
    with ce2:
        lines=[]
        for i,q in enumerate(filtered,1):
            tag = "  ⭐ IMPORTANT" if q.get("important") else ""
            lines += [
                f"Q{i}. [{q.get('difficulty','').upper()}][{q.get('length','').upper()}]{tag}",
                f"Topic: {q.get('topic','')}",
                f"Section: {q.get('section_name','')}" if q.get("section_name") else "",
                f"EN: {q.get('question_en','')}",
                f"BN: {q.get('question_bn','')}" if q.get("question_bn") else "",
                f"\nAnswer (EN): {q.get('answer_en','')}",
                f"Answer (BN): {q.get('answer_bn','')}" if q.get("answer_bn") else "",
                f"Formula: {q.get('formula','')}" if q.get("formula") and q.get("formula")!="null" else "",
                f"Where to find: {q.get('where_to_find','')}" if q.get("where_to_find") else "",
                "\n"+"-"*60+"\n",
            ]
        st.download_button("⬇️ Save as Text",
            data="\n".join(l for l in lines if l),
            file_name="onna_questions.txt", mime="text/plain")

    # ── Question cards ─────────────────────────────────────────
    for i,q in enumerate(filtered):
        diff   = q.get("difficulty","easy").lower()
        length = q.get("length","short")
        imp    = q.get("important",False)
        cls    = "important" if imp else diff
        qtype  = q.get("question_type","")

        badges = ""
        if imp:    badges += '<span class="badge badge-important">⭐ Important</span>'
        badges += f'<span class="badge badge-{diff}">{diff}</span>'
        badges += f'<span class="badge badge-{length}">{length}</span>'
        if qtype:  badges += f'<span class="badge badge-type">{qtype}</span>'

        # ── "Where to find" tags ──────────────────────────────
        find_tags = ""

        # Page reference
        if q.get("page_references"):
            for ref in q["page_references"]:
                find_tags += f'<span class="find-tag"><span class="icon">📌</span><span class="label">Page</span>{ref}</span>'

        # Topic name
        if q.get("topic"):
            clean_topic = q["topic"].split("[p.")[0].strip()  # remove chunk tag
            if clean_topic:
                find_tags += f'<span class="find-tag"><span class="icon">🏷️</span><span class="label">Topic</span>{clean_topic}</span>'

        # Section / chapter name
        if q.get("section_name"):
            find_tags += f'<span class="find-tag"><span class="icon">📑</span><span class="label">Section</span>{q["section_name"]}</span>'

        # Keyword hint
        if q.get("search_keyword"):
            find_tags += f'<span class="find-tag"><span class="icon">🔍</span><span class="label">Search</span>{q["search_keyword"]}</span>'

        # Source file
        if q.get("source_file"):
            find_tags += f'<span class="find-tag"><span class="icon">📄</span><span class="label">File</span>{q["source_file"]}</span>'

        st.markdown(f"""
        <div class="q-card {cls}">
            <div>{badges}</div>
            <div class="q-english">Q{i+1}. {q.get("question_en","")}</div>
            {"<div class='q-bangla'>"+q.get("question_bn","")+"</div>" if q.get("question_bn") else ""}
            <div style="margin-top:8px">{find_tags}</div>
        </div>""", unsafe_allow_html=True)

        with st.expander("💜 Show Answer"):
            if q.get("answer_en"):
                st.markdown('<div style="font-size:10px;letter-spacing:2px;color:#8868a8;'
                            'text-transform:uppercase">English Answer</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="ans-en">{q["answer_en"]}</div>',
                            unsafe_allow_html=True)
            if q.get("formula") and q.get("formula") not in ("null",""):
                st.markdown(f'<div class="formula">{q["formula"]}</div>',
                            unsafe_allow_html=True)
            if q.get("answer_bn"):
                st.markdown('<div style="font-size:10px;letter-spacing:2px;color:#8868a8;'
                            'text-transform:uppercase;margin-top:10px">বাংলা উত্তর</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="ans-bn">{q["answer_bn"]}</div>',
                            unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear & Start Fresh"):
        for k in ["questions","summary","file_info","model_used"]:
            st.session_state[k] = [] if k in ("questions",) else {}
        st.session_state.model_used = None
        st.rerun()
