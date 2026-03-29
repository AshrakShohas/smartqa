# 📚 SmartQA Generator

Generate English + Bangla questions from PDF, DOCX, PPTX files using Google Gemini AI (FREE).

---

## ✅ Requirements
- Python 3.9 or higher
- VS Code (or any terminal)
- Free Google Gemini API key

---

## 🚀 Setup (one time only)

### Step 1 — Get free Gemini API key
1. Go to https://aistudio.google.com
2. Sign in with your Google account
3. Click "Get API Key" → "Create API Key"
4. Copy the key (starts with AIza...)

### Step 2 — Open project in VS Code
```
File → Open Folder → select the "smartqa" folder
```

### Step 3 — Open terminal in VS Code
```
Terminal → New Terminal
```

### Step 4 — Install dependencies (one time)
```bash
pip install -r requirements.txt
```

### Step 5 — Add your API key
Open the `.env` file and replace `paste_your_key_here` with your real key:
```
GEMINI_API_KEY=AIzaSy...your_key_here
```

### Step 6 — Run the app
```bash
streamlit run app.py
```

The app opens automatically in your browser at http://localhost:8501

---

## 📱 Use on Phone (same WiFi)

1. Run the app on your PC as above
2. Find your PC's local IP address:
   - Windows: run `ipconfig` in terminal → look for IPv4 Address (e.g. 192.168.1.5)
   - Mac/Linux: run `ifconfig`
3. On your phone browser, open: `http://192.168.1.5:8501`
   (replace with your actual IP)

OR run with:
```bash
streamlit run app.py --server.address 0.0.0.0
```

---

## ☁️ Deploy FREE online (access from anywhere, any phone)

### Option A — Streamlit Cloud (easiest)
1. Push your code to GitHub (make sure .env is in .gitignore!)
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Add GEMINI_API_KEY in the Secrets section
5. Deploy → get a public URL anyone can open

### Option B — Railway / Render
Both have free tiers. Google "deploy streamlit to Railway" for guides.

---

## 🔑 Free Gemini API Limits
- Model: gemini-1.5-flash
- 15 requests per minute
- 1,500 requests per day
- 1 million tokens per minute
- NO credit card required

---

## 📁 File Structure
```
smartqa/
├── app.py            ← Main Streamlit UI
├── extractor.py      ← PDF/DOCX/PPTX text extraction
├── qa_generator.py   ← Gemini API integration
├── requirements.txt  ← Python dependencies
├── .env              ← Your API key (never share/upload this)
└── README.md         ← This file
```

---

## 🛠️ Customization Tips
- Change model: in `qa_generator.py` line `model = genai.GenerativeModel("gemini-1.5-flash")`
  Try `gemini-1.5-pro` for better quality (lower free limits)
- Change chunk size: adjust `chunk_size` slider in sidebar
- Add more languages: edit the prompt in `qa_generator.py → _build_prompt()`
- Change UI colors: edit the CSS in `app.py` inside `st.markdown("""<style>...</style>""")`
