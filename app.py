import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vidora AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "theme": "dark",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Theme Definitions (ChatGPT-style) ───────────────────────────────────────────
THEMES = {
    "dark": {
        "bg": "#212121",
        "sidebar": "#171717",
        "surface": "#2f2f2f",
        "surface-2": "#3a3a3a",
        "border": "#4d4d4f",
        "text": "#ececec",
        "text-muted": "#a6a6a6",
        "accent": "#10a37f",
        "accent-hover": "#1a7f64",
        "user-bubble": "#2f2f2f",
        "bot-bubble": "transparent",
        "input-bg": "#2f2f2f",
    },
    "light": {
        "bg": "#ffffff",
        "sidebar": "#f7f7f8",
        "surface": "#f7f7f8",
        "surface-2": "#ececec",
        "border": "#e3e3e3",
        "text": "#0d0d0d",
        "text-muted": "#6e6e80",
        "accent": "#10a37f",
        "accent-hover": "#0d8a6c",
        "user-bubble": "#f4f4f4",
        "bot-bubble": "transparent",
        "input-bg": "#ffffff",
    },
}

t = THEMES[st.session_state.theme]

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Söhne:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --bg: {t['bg']};
    --sidebar: {t['sidebar']};
    --surface: {t['surface']};
    --surface-2: {t['surface-2']};
    --border: {t['border']};
    --text: {t['text']};
    --text-muted: {t['text-muted']};
    --accent: {t['accent']};
    --accent-hover: {t['accent-hover']};
    --user-bubble: {t['user-bubble']};
    --bot-bubble: {t['bot-bubble']};
    --input-bg: {t['input-bg']};
}}

html, body, [class*="css"] {{
    font-family: 'Inter', 'Söhne', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}}

.stApp {{
    background: var(--bg) !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
}}

[data-testid="stSidebar"] * {{
    color: var(--text) !important;
}}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}}

/* ── Hero / Brand ── */
.hero-title {{
    font-family: 'Inter', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.2;
    margin: 0;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.hero-sub {{
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}}

.brand-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px; height: 32px;
    border-radius: 8px;
    background: var(--accent);
    color: white;
    font-weight: 700;
    font-size: 1rem;
}}

/* ── Cards (ChatGPT panel style) ── */
.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}}

.card-title {{
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.card-content {{
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text);
}}

/* ── Badges ── */
.badge {{
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}}

.badge-purple {{ background: rgba(16,163,127,0.15); color: var(--accent); border: 1px solid rgba(16,163,127,0.3); }}
.badge-cyan   {{ background: rgba(59,130,246,0.12); color: #3b82f6;      border: 1px solid rgba(59,130,246,0.3); }}
.badge-green  {{ background: rgba(16,163,127,0.15); color: var(--accent); border: 1px solid rgba(16,163,127,0.3); }}

/* ── Inputs & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {{
    background: var(--input-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}}

.stTextInput > div > div > input:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(16,163,127,0.2) !important;
}}

.stButton > button {{
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.15s !important;
}}

.stButton > button:hover {{
    background: var(--accent-hover) !important;
    transform: translateY(-1px) !important;
}}

.stButton > button[kind="secondary"] {{
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}}

/* ── Progress / Status ── */
.status-bar {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.9rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.35rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
}}

.status-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}}

.dot-active   {{ background: var(--accent); box-shadow: 0 0 8px var(--accent); animation: pulse 1.5s infinite; }}
.dot-done     {{ background: var(--accent); }}
.dot-pending  {{ background: var(--border); }}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.4; }}
}}

/* ── ChatGPT-style Chat ── */
.chat-container {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0;
    max-height: 480px;
    overflow-y: auto;
    margin-bottom: 1rem;
}}

.chat-msg {{
    padding: 1rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    border-bottom: 1px solid var(--border);
}}

.chat-msg.user-msg {{
    background: var(--user-bubble);
}}

.chat-msg.bot-msg {{
    background: var(--bot-bubble);
}}

.chat-label {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

.chat-bubble {{
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text);
}}

.user-label  {{ color: var(--accent); }}
.bot-label   {{ color: #3b82f6; }}

/* ── Divider ── */
hr {{
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}}

/* ── Transcript box ── */
.transcript-box {{
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.85rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
    font-family: 'JetBrains Mono', monospace;
}}

/* ── Stale Streamlit elements ── */
.stProgress > div > div > div {{ background: var(--accent) !important; }}
.stSpinner > div {{ border-top-color: var(--accent) !important; }}
[data-testid="stMarkdownContainer"] p {{ color: var(--text) !important; }}
label {{ color: var(--text-muted) !important; font-size: 0.8rem !important; }}

/* scrollbar */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    return "dot-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="hero-title"><span class="brand-badge">V</span> Vidora AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-sub">Video Intelligence by Creatix</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Theme toggle
    theme_choice = st.selectbox(
        "Theme",
        ["dark", "light"],
        index=0 if st.session_state.theme == "dark" else 1,
        format_func=lambda x: "🌙 Dark" if x == "dark" else "☀️ Light",
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Extraction"),
            ("rag",        "🧠", "RAG Engine"),
        ]:
            render_step_bar(label, step, icon)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title"><span class="brand-badge">V</span> Vidora AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            with progress_placeholder.container():
                st.info("⚙️ Pipeline running — see sidebar for live status…")

            update_step("audio", "active")
            chunks = process_input(source)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items  = extract_action_items(transcript)
            decisions     = extract_key_decisions(transcript)
            questions     = extract_questions(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            progress_placeholder.success("✅ Analysis complete!")
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio","transcript","title","summary","extract","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"❌ Error: {e}")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Title banner
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Inter',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {r['title']}
        </div>
    </div>""", unsafe_allow_html=True)

    # Top row: summary + transcript
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)

    # Second row: action items | decisions | questions
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Key Decisions</div>
            <div class="card-content">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">❓ Open Questions</div>
            <div class="card-content">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Inter\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>', unsafe_allow_html=True)

    # Chat history display
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg user-msg">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble">{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg bot-msg">
                    <span class="chat-label bot-label">✨ Vidora AI</span>
                    <div class="chat-bubble">{msg['content']}</div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
            <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>
        </div>""", unsafe_allow_html=True)

    # Chat input
    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input("Your question", placeholder="What were the main decisions made?", label_visibility="collapsed")
    with chat_col2:
        send_btn = st.button("Send →", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">✨</div>
        <div style="font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Paste a YouTube URL or local file path in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)