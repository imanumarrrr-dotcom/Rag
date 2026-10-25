# DocMind - PDF RAG Agent (Streamlit). Run: python -m streamlit run rag.py
import hashlib, html, os, tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AuthenticationError, OpenAI

if "_DOCMIND_STARTED" in globals():  # file pasted twice? ignore the second copy instead of crashing
    st.stop()
_DOCMIND_STARTED = True

st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

GRAD = "linear-gradient(90deg,#F5C344,#F28482,#B567C2)"
CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap");
@keyframes rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes shimmer{to{background-position:200% center}}
@keyframes cursor{0%,15%{transform:translate(0,0)}45%,60%{transform:translate(-62px,-22px)}100%{transform:translate(0,0)}}
@keyframes press{0%,40%,70%,100%{transform:scale(1)}50%,60%{transform:scale(.93);box-shadow:0 0 0 8px rgba(168,85,247,.18)}}
@keyframes caret{50%{opacity:0}}
@keyframes dots{0%,80%,100%{transform:scale(.5);opacity:.3}40%{transform:scale(1);opacity:1}}
@keyframes glowbg{0%,100%{background-position:50% 0}50%{background-position:50% 14%}}
@keyframes step{from{opacity:.4}to{opacity:1;border-color:#B567C2;background:#faf5ff}}
html,body,.stApp{background:#fff!important;font-family:Inter,system-ui,sans-serif;color:#0f172a}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stAppDeployButton"],#MainMenu,[data-testid="stMainMenu"]{display:none!important}
.block-container{max-width:1100px!important;padding:3rem 20px 7rem}
[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:transparent!important}
.stApp p,.stApp h1,.stApp h2,.stApp h3,.stApp label,.stApp input,.stApp textarea,.stApp li{font-family:Inter,system-ui,sans-serif}
[data-testid="stIconMaterial"],.material-symbols-rounded,.material-icons{font-family:"Material Symbols Rounded","Material Icons"!important;font-feature-settings:"liga"}
.stApp [data-testid="stMarkdownContainer"] p,.stApp [data-testid="stMarkdownContainer"] li,.stApp label,.stApp [data-testid="stWidgetLabel"] *{color:#1e293b!important}
.stApp [data-testid="stCaptionContainer"] *{color:#64748b!important}
.badge{font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;background:GRAD;background-size:200% auto;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:shimmer 6s linear infinite;margin-bottom:16px;text-align:center}
.title{font-size:2.75rem;font-weight:500;color:#0f172a;letter-spacing:-.02em;margin-bottom:12px;text-align:center;animation:rise .8s .1s both}
.subtitle{font-size:1.125rem;color:#64748b;line-height:1.5;margin-bottom:40px;text-align:center;animation:rise .8s .25s both}
/* login form */
[data-testid="stForm"]{max-width:440px;margin:0 auto 56px;background:#fff;border:1px solid #e2e8f0;border-radius:20px;padding:1.6rem;box-shadow:0 10px 30px -10px rgba(0,0,0,.12);animation:rise .8s .4s both}
.stApp [data-baseweb="input"],.stApp [data-baseweb="select"]>div{background:#F4F8F9!important;border:1px solid #cbd5e1!important;border-radius:12px!important}
.stApp [data-baseweb="base-input"]{background:transparent!important}
.stApp input{background:transparent!important;color:#0f172a!important;-webkit-text-fill-color:#0f172a!important}
.stApp [data-baseweb="select"] *{color:#0f172a!important}
[data-testid="stTextInputRootElement"] button{background:transparent!important;color:#64748b!important}
.stApp .stButton>button,.stApp [data-testid="stFormSubmitButton"]>button{background:GRAD;background-size:200% auto;border:0;border-radius:12px;font-weight:600;min-height:2.8rem;color:#0f172a!important;transition:transform .2s,box-shadow .2s,background-position .6s}
.stApp .stButton>button *,.stApp [data-testid="stFormSubmitButton"]>button *{color:#0f172a!important}
.stApp .stButton>button:hover,.stApp [data-testid="stFormSubmitButton"]>button:hover{transform:translateY(-2px);box-shadow:0 12px 26px -8px rgba(181,103,194,.55);background-position:100% center}
/* cards */
.c1-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}
@media(max-width:900px){.c1-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:600px){.c1-grid{grid-template-columns:1fr}.title{font-size:2.25rem}}
.c1-card{border-radius:20px;height:340px;display:flex;flex-direction:column;justify-content:flex-end;position:relative;overflow:hidden;text-align:left;background:#F4F8F9;box-shadow:0 10px 30px -10px rgba(0,0,0,.1);animation:rise .8s both;transition:transform .3s,box-shadow .3s}
.c1-card:hover{transform:translateY(-6px);box-shadow:0 22px 44px -14px rgba(0,0,0,.18)}
.c1-card h3{font-size:1.05rem;font-weight:600;color:#1e293b;padding:24px;z-index:2;margin:0}
.c1-card-1{background:radial-gradient(circle at 50% 0%,#FFB347 0%,#F9ED96 30%,#F4F8F9 60%,#F4F8F9 100%);background-size:100% 130%;animation:rise .8s .55s both,glowbg 7s ease-in-out infinite}
.c1-card-2{background:radial-gradient(circle at 50% 0%,#E5A1F5 0%,#F8ACA0 30%,#F4F8F9 60%,#F4F8F9 100%);background-size:100% 130%;animation:rise .8s .7s both,glowbg 8s ease-in-out infinite}
.c1-card-3{background:radial-gradient(circle at 50% 0%,#F9ED96 0%,#E5A1F5 30%,#F4F8F9 60%,#F4F8F9 100%);background-size:100% 130%;animation:rise .8s .85s both,glowbg 9s ease-in-out infinite}
.pbox{position:absolute;top:30px;left:24px;right:24px;background:#fff;border-radius:12px;padding:16px;font-size:.8rem;color:#475569;line-height:1.6;box-shadow:0 8px 20px rgba(0,0,0,.04)}
.pbox::after{content:"";display:inline-block;width:2px;height:.9em;background:#a855f7;margin-left:2px;vertical-align:-2px;animation:caret 1s steps(1) infinite}
.blur{font-weight:600;background:linear-gradient(90deg,#FFB347,#E5A1F5,#FFB347);background-size:200% auto;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:shimmer 4s linear infinite}
.pill{position:absolute;background:#fff;border:1px solid #000;border-radius:20px;padding:5px 14px;font-size:.75rem;font-weight:600;color:#1e293b;box-shadow:0 4px 15px rgba(0,0,0,.08);white-space:nowrap;display:flex;gap:6px;align-items:center}
.pill1{top:180px;left:40px;animation:press 5s ease-in-out infinite}.pill1 i{font-style:normal;color:#a855f7;font-size:1rem}
.cur{position:absolute;top:205px;left:110px;width:24px;height:24px;background:#0f172a;clip-path:polygon(17% 8%,83% 46%,46% 54%,38% 92%);z-index:10;filter:drop-shadow(0 4px 6px rgba(0,0,0,.2));animation:cursor 5s ease-in-out infinite}
.vis{position:absolute;top:0;left:0;right:0;bottom:70px;display:flex;align-items:center;justify-content:center;padding:0 24px}
.vis img{width:100%;height:180px;object-fit:contain;margin-top:20px;animation:floaty 5s ease-in-out infinite}
.mesh{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.8) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.8) 1px,transparent 1px);background-size:16px 16px;-webkit-mask-image:radial-gradient(circle at center top,black 0%,transparent 80%);mask-image:radial-gradient(circle at center top,black 0%,transparent 80%)}
.folder{position:absolute;top:50px;left:calc(50% - 85px);width:170px;filter:drop-shadow(0 15px 25px rgba(0,0,0,.08));animation:floaty 6s ease-in-out infinite}
.search{position:absolute;top:220px;left:50%;transform:translateX(-50%);background:#fff;border:1px solid #000;border-radius:20px;padding:6px 18px;font-size:.75rem;font-weight:500;color:#1e293b;box-shadow:0 8px 20px rgba(0,0,0,.06);white-space:nowrap;display:flex;gap:8px;align-items:center}
/* chat */
[data-testid="stChatMessageAvatarUser"],[data-testid="stChatMessageAvatarAssistant"]{display:none!important}
[data-testid="stChatMessage"]{background:transparent;animation:rise .45s both;line-height:1.7}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *{color:#1e293b!important}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){width:fit-content;max-width:86%;margin-left:auto;background:linear-gradient(135deg,#FFF6DA,#F9E3F7);border:1px solid #e9d5ff;border-radius:20px 20px 6px 20px;padding:.8rem 1.1rem}
[data-testid="stChatInput"]>div{background:#fff!important;border:1px solid #cbd5e1!important;border-radius:28px!important;box-shadow:0 10px 30px -10px rgba(0,0,0,.15)}
[data-testid="stChatInput"]>div:focus-within{border-color:#B567C2!important;box-shadow:0 0 0 4px rgba(181,103,194,.18)}
[data-testid="stChatInput"] textarea{color:#0f172a!important;-webkit-text-fill-color:#0f172a!important}
[data-testid="stChatInput"] button{background:GRAD!important;color:#0f172a!important;border-radius:50%!important}
.thinking{display:flex;gap:7px;padding:.7rem 0}.thinking span{width:9px;height:9px;border-radius:50%;background:GRAD;animation:dots 1.2s infinite}
.thinking span:nth-child(2){animation-delay:.15s}.thinking span:nth-child(3){animation-delay:.3s}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:.6rem 0}.chip{border:1px solid #e2e8f0;background:#fff;border-radius:999px;padding:.15rem .8rem;font-size:.85rem;color:#1e293b}
.pipe{display:flex;gap:10px;margin:1rem 0;flex-wrap:wrap}.pipe span{flex:1;min-width:120px;text-align:center;padding:.7rem;border-radius:14px;border:1px solid #e2e8f0;background:#fff;opacity:.4;font-weight:500;animation:step .5s forwards}
.pipe span:nth-child(1){animation-delay:.2s}.pipe span:nth-child(2){animation-delay:1.6s}.pipe span:nth-child(3){animation-delay:3.2s}
[data-testid="stFileUploader"] section{background:radial-gradient(circle at 50% 0%,#F9ED96 0%,#F4F8F9 55%);border:2px dashed #B567C2;border-radius:20px;padding:2rem;transition:transform .2s}
[data-testid="stFileUploader"] section:hover{transform:translateY(-3px)}
[data-testid="stFileUploader"] section *{color:#1e293b!important}
[data-testid="stFileUploader"] button{background:GRAD!important;border:0!important;border-radius:10px!important}
[data-testid="stSidebar"]{background:#F4F8F9;border-right:1px solid #e2e8f0}
.brand{font-size:1.4rem;font-weight:600;color:#0f172a;margin-bottom:1rem}
[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapseButton"] *{opacity:1!important;color:#0f172a!important}

/* ---- polish: expander, dropdown, uploader, actions, messages ---- */
[data-testid="stExpander"]{border:1px solid #e2e8f0!important;border-radius:14px!important;background:#fff;box-shadow:0 6px 18px -10px rgba(0,0,0,.12)}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:#1e293b!important;font-weight:500}
[data-testid="stExpander"] [data-testid="stText"],[data-testid="stExpander"] pre{color:#334155!important;background:#F4F8F9!important;border-radius:10px;padding:.6rem .8rem;white-space:pre-wrap}
[data-baseweb="popover"] *,[data-baseweb="menu"] *{color:#0f172a!important}
[data-baseweb="popover"] [data-baseweb="menu"],[data-baseweb="popover"]>div{background:#fff!important;border-radius:12px}
[data-baseweb="popover"] li:hover{background:#faf5ff!important}
[data-testid="stFileUploader"] section{padding:1.1rem 1.4rem!important}
[data-testid="stFileUploaderFile"] *{color:#1e293b!important}
[data-testid="stFileUploader"] button[kind]{min-height:2.4rem}
.stApp [class*="st-key-act_"] button,.stApp [class*="st-key-side_"] button{background:#fff!important;border:1px solid #e9d5ff!important;box-shadow:0 6px 16px -10px rgba(168,85,247,.5)}
.stApp [class*="st-key-act_"] button:hover,.stApp [class*="st-key-side_"] button:hover{background:linear-gradient(135deg,#FFF6DA,#F9E3F7)!important;border-color:#B567C2!important}
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])){border-left:3px solid #E5A1F5;padding-left:1rem;margin:.6rem 0}
[data-testid="stSlider"] [role="slider"]{box-shadow:0 0 0 4px rgba(181,103,194,.2)}
[data-testid="stToggle"] label *{color:#1e293b!important}
.stApp code{background:#F4F8F9!important;color:#0f172a!important;border-radius:6px}

/* ---- workspace design ---- */
@keyframes aurora{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
@keyframes borderflow{to{background-position:0 0,300% 0}}
@keyframes drift{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(5vw,4vh) scale(1.15)}}
@keyframes sparkle{0%,100%{opacity:.35;transform:scale(.8) rotate(0)}50%{opacity:1;transform:scale(1.2) rotate(25deg)}}
@keyframes ring{0%{box-shadow:0 0 0 0 rgba(181,103,194,.35)}100%{box-shadow:0 0 0 18px rgba(181,103,194,0)}}
.stApp::before{content:"";position:fixed;inset:-10%;z-index:0;pointer-events:none;animation:drift 26s ease-in-out infinite;
 background:radial-gradient(34vw 34vw at 12% 18%,rgba(255,226,170,.55),transparent 62%),radial-gradient(32vw 32vw at 88% 70%,rgba(229,161,245,.40),transparent 62%)}
[data-testid="stMain"]{position:relative;z-index:1}
.wshero{position:relative;overflow:hidden;border-radius:24px;padding:34px 40px;margin-bottom:26px;
 background:linear-gradient(120deg,#FFE9B8,#FBD1D0,#EBD2F7,#FFE9B8);background-size:300% 300%;
 animation:aurora 9s ease-in-out infinite,rise .7s both;box-shadow:0 20px 50px -20px rgba(181,103,194,.45)}
.wshero .badge{text-align:left;margin-bottom:10px}
.wtitle{font-size:2.6rem;font-weight:500;letter-spacing:-.02em;color:#0f172a}
.wsub{font-size:1.1rem;color:#475569;margin-top:6px}
.sp{position:absolute;font-style:normal;color:#a855f7;animation:sparkle 3.5s ease-in-out infinite}
.sp.s1{right:9%;top:22%;font-size:1.8rem}.sp.s2{right:20%;top:58%;font-size:1.1rem;animation-delay:1s}.sp.s3{right:4%;top:62%;font-size:1.3rem;animation-delay:2s}
.stats{display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:14px;margin:18px 0 14px}
@media(max-width:700px){.stats{grid-template-columns:1fr 1fr}}
.stat{background:#fff;border:1px solid #eef0f4;border-radius:16px;padding:14px 18px;box-shadow:0 10px 24px -16px rgba(0,0,0,.2);animation:rise .6s both;transition:transform .25s,box-shadow .25s}
.stat:hover{transform:translateY(-4px);box-shadow:0 18px 30px -16px rgba(181,103,194,.5)}
.stat:nth-child(2){animation-delay:.08s}.stat:nth-child(3){animation-delay:.16s}.stat:nth-child(4){animation-delay:.24s}
.stat small{display:block;color:#64748b;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;font-weight:600}
.stat b{display:block;font-size:1.35rem;font-weight:600;background:GRAD;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
[data-testid="stFileUploader"] section{background:linear-gradient(#fff,#fff) padding-box,linear-gradient(90deg,#F5C344,#F28482,#B567C2,#F5C344) border-box!important;background-size:100% 100%,300% 100%!important;border:2px solid transparent!important;border-radius:20px!important;animation:borderflow 6s linear infinite}
.welcome{display:flex;gap:16px;align-items:center;background:#fff;border:1px solid #eef0f4;border-radius:20px;padding:18px 22px;margin:12px 0;box-shadow:0 14px 30px -18px rgba(0,0,0,.25);animation:rise .6s .3s both}
.welcome .orb{flex:none;width:44px;height:44px;border-radius:50%;background:GRAD;background-size:200% auto;animation:ring 2s ease-out infinite,shimmer 5s linear infinite}
.welcome b{display:block;color:#0f172a}.welcome span{color:#64748b}
.stApp .stButton>button p{font-family:Inter,system-ui,sans-serif}
.brand{background:GRAD;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;font-weight:600}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}
</style>
""".replace("GRAD", GRAD)
st.markdown(CSS, unsafe_allow_html=True)

CARDS = """
<div class="c1-grid">
 <div class="c1-card c1-card-1">
  <div class="pbox">What is the refund policy in this <span class="blur">contract</span>? Summarize the <span class="blur">key terms</span> and <span class="blur">cite the pages</span></div>
  <div class="pill pill1"><i>✦</i> Add more details</div><div class="cur"></div>
  <h3>Ask in Plain Language</h3></div>
 <div class="c1-card c1-card-2">
  <div class="vis"><img src="https://pub-f170a2592d2c4a1485466404c36807be.r2.dev/viktor/network.svg" alt="" onerror="this.style.display='none'"></div>
  <h3>Your Key, Your Control</h3></div>
 <div class="c1-card c1-card-3">
  <div class="mesh"></div><img class="folder" src="https://pub-f170a2592d2c4a1485466404c36807be.r2.dev/viktor/library%20icon.svg" alt="" onerror="this.style.display='none'">
  <div class="search">🔍 Search in your PDF</div>
  <h3>Answers With Page Citations</h3></div>
</div>
"""


# ------------------------------------------------------------------ RAG logic (unchanged idea)
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer questions using only the supplied PDF context. "
               "If the user asks what the document is about, asks for a summary, key points or suggested questions, "
               "base it on the context. If the answer is not present, say: \"I couldn't find that in the document.\" "
               "Cite the page numbers used, for example (p. 3). Use short paragraphs or bullet points.\n\n"
               "Conversation so far:\n{history}\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def build_vectorstore(pdf_bytes, size, overlap, api_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name
    try:
        docs = PyPDFLoader(path).load()
    finally:
        if os.path.exists(path):
            os.unlink(path)
    if not docs:
        raise ValueError("No readable pages were found in the PDF.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_documents(docs)
    if not chunks:
        raise ValueError("No text could be extracted from the PDF.")
    emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)  # key passed explicitly, never in os.environ
    return FAISS.from_documents(chunks, emb), len(docs), len(chunks)


def validate_key(key):
    try:
        OpenAI(api_key=key, max_retries=0, timeout=20).models.list()
    except AuthenticationError:
        return "Invalid API key. Check it and try again."
    except Exception as e:  # noqa: BLE001
        return f"Could not validate the key: {e}"
    return None


# ------------------------------------------------------------------ login screen
if not st.session_state.get("authed"):
    st.markdown('<div class="badge">PDF RAG Agent</div><div class="title">Ask your PDFs anything</div>'
                '<div class="subtitle">Sign in with your OpenAI key and go<br>from document to answer</div>', unsafe_allow_html=True)
    with st.form("login"):
        key = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
        go = st.form_submit_button("Enter", use_container_width=True)
        st.caption("Your key stays in this session's memory only. It is never saved to disk.")
    if go:
        if not key.strip():
            st.error("Please enter your OpenAI API key.")
        else:
            with st.spinner("Checking your API key..."):
                err = validate_key(key.strip())
            if err:
                st.error(err)
            else:
                st.session_state.authed, st.session_state.api_key = True, key.strip()
                st.rerun()
    st.markdown(CARDS, unsafe_allow_html=True)
    st.stop()

api_key = st.session_state.api_key

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="brand">📄 DocMind</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], key="dm_model")
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100, key="dm_cs")
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50, key="dm_co")
    top_k = st.slider("Retrieved chunks (k)", 2, 10, 4, key="dm_k")
    use_mmr = st.toggle("Diverse results (MMR)", value=True, key="dm_mmr", help="Picks varied chunks instead of near-duplicates.")
    st.button("Logout", key="dm_logout", on_click=lambda: st.session_state.clear(), use_container_width=True)

# ------------------------------------------------------------------ main
st.markdown('<div class="wshero"><i class="sp s1">✦</i><i class="sp s2">✦</i><i class="sp s3">✦</i>'
            '<div class="badge">Workspace</div><div class="wtitle">Your PDF, answered.</div>'
            '<div class="wsub">Upload a document, then ask anything or tap a quick action.</div></div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop a PDF here", type=["pdf"], key="dm_pdf")
if uploaded is None:
    st.markdown('<div style="height:26px"></div>' + CARDS, unsafe_allow_html=True)
    st.stop()

pdf_bytes = uploaded.getvalue()
sig = (hashlib.sha256(pdf_bytes).hexdigest(), chunk_size, chunk_overlap)
if st.session_state.get("file_sig") != sig:
    slot = st.empty()
    slot.markdown('<div class="pipe"><span>1 · Reading</span><span>2 · Splitting</span><span>3 · Embedding</span></div>', unsafe_allow_html=True)
    try:
        vs, pages, n = build_vectorstore(pdf_bytes, chunk_size, chunk_overlap, api_key)
    except Exception as e:  # noqa: BLE001
        slot.empty()
        st.error(f"PDF indexing failed: {e}")
        st.stop()
    slot.empty()
    st.session_state.update(vectorstore=vs, file_sig=sig, messages=[], stats=(pages, n))

pages, n = st.session_state.stats
st.markdown(f'<div class="stats"><div class="stat"><small>Document</small><b>{html.escape(uploaded.name)}</b></div>'
            f'<div class="stat"><small>Pages</small><b>{pages}</b></div><div class="stat"><small>Chunks</small><b>{n}</b></div>'
            f'<div class="stat"><small>Model</small><b>{html.escape(model_name)}</b></div></div>', unsafe_allow_html=True)


def show_sources(srcs):
    pg = sorted({s["page"] for s in srcs})
    st.markdown('<div class="chips">' + "".join(f'<span class="chip">p. {p}</span>' for p in pg) + "</div>", unsafe_allow_html=True)
    with st.expander("🔍 Retrieved sources"):
        for s in srcs:
            st.markdown(f"**Page {s['page']}**")
            st.text(s["text"])


ACTIONS = [("📝 Summarize", "Summarize this document in a few clear paragraphs."),
           ("🔑 Key points", "List the key points of this document as bullet points with page numbers."),
           ("❓ Suggest questions", "Suggest 5 good questions I could ask about this document.")]
for col, (i, (label, q)) in zip(st.columns(len(ACTIONS)), enumerate(ACTIONS)):
    if col.button(label, key=f"act_{i}", use_container_width=True):
        st.session_state.queued = (q, True)

if not st.session_state.messages:
    st.markdown('<div class="welcome"><div class="orb"></div><div><b>Your document is ready.</b>'
                '<span>Ask a question below, or tap Summarize, Key points or Suggest questions.</span></div></div>', unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

typed = st.chat_input("Ask a question about your PDF...")
queued = st.session_state.pop("queued", None)
question, wide = (typed, False) if typed else (queued or (None, False))
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.chat_message("assistant"):
            dots = st.empty()
            dots.markdown('<div class="thinking"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
            vs = st.session_state.vectorstore
            if wide:  # summaries need broad context, not just the nearest chunks
                docs = list(vs.docstore._dict.values())[:12]
            elif use_mmr:
                docs = vs.max_marginal_relevance_search(question, k=top_k, fetch_k=top_k * 4)
            else:
                docs = vs.similarity_search(question, k=top_k)
            context = "\n\n".join(f"[p. {d.metadata.get('page', 0) + 1}] {d.page_content}" for d in docs)
            history = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in st.session_state.messages[-7:-1]) or "(none)"
            llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)

            def tokens():
                first = True
                for c in llm.stream(RAG_PROMPT.format_messages(context=context, question=question, history=history)):
                    if c.content:
                        if first:
                            dots.empty()
                            first = False
                        yield c.content

            answer = st.write_stream(tokens())
            srcs = [{"page": d.metadata.get("page", 0) + 1, "text": d.page_content[:500]} for d in docs]
            show_sources(srcs)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": srcs})
    except Exception as e:  # noqa: BLE001
        st.session_state.messages.pop()
        st.error(f"Question failed: {e}")

# chat tools (added last so the download always contains the newest answer)
with st.sidebar:
    st.divider()
    st.button("🧹 Clear chat", key="side_clear", on_click=lambda: st.session_state.update(messages=[]), use_container_width=True)
    st.download_button("⬇️ Download chat", "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages) or "(empty)",
                       file_name="chat.txt", key="side_dl", use_container_width=True)