# Adnan-ATS-resume-
# 📄 ATS Resume Analyzer

Upload a resume (PDF, DOCX or TXT) and get an **ATS score** plus specific, prioritized
**improvements**, missing keywords and rewritten bullet points.

Built with **Streamlit** (UI) and the **Groq API** (AI review).

## How the score works

The final score blends two independent signals:

| Part | Weight | What it checks |
|------|--------|----------------|
| Rule-based checks | 40% | contact info, standard sections, length, quantified achievements, action verbs, parse-friendly characters, and keyword match against an optional job description |
| AI review (Groq) | 60% | content quality, impact & metrics, keyword relevance, structure, clarity |

If the AI call fails (bad key, rate limit), the app still returns the rule-based score.

## Features

- PDF / DOCX / TXT upload (clear errors for scanned or password-protected PDFs)
- Optional job description for targeted keyword matching
- Prioritized improvements (high / medium / low)
- Strengths, missing keywords, before/after bullet rewrites
- Downloadable text report

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Get a free API key at <https://console.groq.com/keys>. You can either paste it in the app's
sidebar, or set it as an environment variable:

```bash
export GROQ_API_KEY="gsk_..."   # Windows PowerShell: $env:GROQ_API_KEY="gsk_..."
```

For local secrets you can also create `.streamlit/secrets.toml` (**never commit this file**):

```toml
GROQ_API_KEY = "gsk_..."
```

## Choosing a model

The default is `openai/gpt-oss-20b` (fastest self-serve model on Groq). `openai/gpt-oss-120b` is
available in the sidebar for higher-quality reviews. Groq changes its model lineup from time to time -
check <https://console.groq.com/docs/models> and edit `MODEL_OPTIONS` in `app.py` if needed.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public, or private with access granted).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → choose the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add: `GROQ_API_KEY = "gsk_your_key_here"`
5. Click **Deploy**.

## Project structure

```
app.py             # Streamlit app (extraction, scoring, Groq call, UI)
requirements.txt   # Python dependencies
README.md
```

## Privacy

Resume text is sent to Groq for analysis and is not stored by this app. Don't upload resumes
you aren't comfortable sending to a third-party API.

## Limitations

- ATS systems differ; the score is a helpful estimate, not a guarantee.
- Scanned/image-only resumes can't be read (and neither can most ATS software).
- AI suggestions can be wrong - never add skills or experience you don't actually have.
