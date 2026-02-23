# app.py
# Streamlit Cloud-ready: St. Johns Girls Secondary School FAQ Chatbot (Kaloleni, Kilifi County)
# - Uses OpenAI if OPENAI_API_KEY is set in Streamlit Secrets or environment
# - Falls back to offline matching if OpenAI is unavailable
#
# Local run:
#   pip install -r requirements.txt
#   streamlit run app.py

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st

# OpenAI SDK (official)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# ----------------------------
# SCHOOL CONFIG
# ----------------------------
SCHOOL_NAME = "St. Johns Girls Secondary School"
SCHOOL_LOCATION = "Kaloleni, Kilifi County"
SCHOOL_PHONE = "0721815047"
SCHOOL_EMAIL = "johnspekee@gmail.com"

DEFAULT_FALLBACK = (
    "Sorry — I’m not sure about that yet. "
    f"Please contact the school office: 📞 {SCHOOL_PHONE} • 📧 {SCHOOL_EMAIL}"
)

# ----------------------------
# FAQ DATA (EDIT/EXPAND THIS)
# ----------------------------
FAQS: List[Dict[str, object]] = [
    {
        "id": "admissions",
        "questions": [
            "How do I apply?",
            "How can I join the school?",
            "Admissions process",
            "How to register",
            "What documents are needed for admission?",
        ],
        "answer": (
            "Admissions are handled through the school office. Collect an application form, "
            "submit it with the required documents, and follow the instructions provided by the office."
        ),
        "tags": ["admissions", "apply", "register", "join", "documents"],
    },
    {
        "id": "fees",
        "questions": [
            "How much are school fees?",
            "What is the fee structure?",
            "When do we pay fees?",
            "Fees payment",
            "How do I pay fees?",
        ],
        "answer": (
            "Fees depend on the class/form and term. Please confirm the latest fee structure "
            "from the bursar's office."
        ),
        "tags": ["fees", "payment", "bursar"],
    },
    {
        "id": "term_dates",
        "questions": [
            "When does the term start?",
            "When does the term end?",
            "School calendar",
            "Opening date",
            "Closing date",
        ],
        "answer": (
            "Term dates are shared in the school calendar. If you provide the term name (e.g., Term 1), "
            "the office or class teacher can confirm the exact opening and closing dates."
        ),
        "tags": ["term", "dates", "calendar", "opening", "closing"],
    },
    {
        "id": "uniform",
        "questions": [
            "What is the school uniform?",
            "Uniform requirements",
            "Where to buy uniform",
        ],
        "answer": (
            "The uniform includes the official school items (approved dress/skirt, blouse/shirt, sweater, "
            "and black shoes). Confirm the exact uniform list and supplier through the school office."
        ),
        "tags": ["uniform", "clothes", "shoes"],
    },
    {
        "id": "reporting_time",
        "questions": [
            "What time should students report?",
            "When does school start each day?",
            "Reporting time",
            "Morning assembly time",
        ],
        "answer": (
            "Students should report before morning assembly. For the exact reporting time, please confirm "
            "with the school office or your class teacher."
        ),
        "tags": ["reporting", "time", "start", "assembly"],
    },
    {
        "id": "contact",
        "questions": [
            "How can I contact the school?",
            "School phone number",
            "School email",
            "Contact details",
            "Where is the school located?",
            "What is the phone number?",
            "What is the email address?",
        ],
        "answer": (
            f"{SCHOOL_NAME} is located in {SCHOOL_LOCATION}.\n\n"
            f"📞 Phone: {SCHOOL_PHONE}\n"
            f"📧 Email: {SCHOOL_EMAIL}\n\n"
            "You can call or email the school office during working hours for assistance."
        ),
        "tags": ["contact", "phone", "email", "office", "location", "number"],
    },
]

# ----------------------------
# OFFLINE MATCHER (fallback)
# ----------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "for", "in", "on", "at", "is", "are",
    "was", "were", "be", "been", "being", "i", "we", "you", "they", "he", "she", "it",
    "this", "that", "these", "those", "my", "our", "your", "their", "me", "us",
    "please", "kindly", "tell", "about", "can", "could", "would", "how", "what", "when",
    "where", "who", "why"
}

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    text = normalize(text)
    return [t for t in text.split(" ") if t and t not in STOPWORDS]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def build_question_index(faqs: List[Dict[str, object]]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for f in faqs:
        fid = str(f["id"])
        for q in f["questions"]:
            items.append((fid, str(q)))
    return items

QUESTION_INDEX = build_question_index(FAQS)

def match_rule_based(user_text: str, faqs: List[Dict[str, object]]) -> Optional[str]:
    text = normalize(user_text)
    for f in faqs:
        tags = [normalize(t) for t in f.get("tags", [])]
        for tag in tags:
            if re.search(rf"(^|\s){re.escape(tag)}(\s|$)", text):
                return str(f["id"])
    return None

@dataclass
class MatchResult:
    faq_id: str
    score: float
    matched_question: str

def match_similarity(user_text: str) -> Optional[MatchResult]:
    u_tokens = tokenize(user_text)
    best_score = 0.0
    best_faq = None
    best_q = None
    for fid, q in QUESTION_INDEX:
        score = jaccard(u_tokens, tokenize(q))
        if score > best_score:
            best_score = score
            best_faq = fid
            best_q = q
    if best_faq is None:
        return None
    return MatchResult(faq_id=best_faq, score=best_score * 100.0, matched_question=str(best_q))

def get_answer_by_id(faq_id: str, faqs: List[Dict[str, object]]) -> str:
    for f in faqs:
        if str(f["id"]) == str(faq_id):
            return str(f["answer"])
    return DEFAULT_FALLBACK

def offline_reply(user_text: str) -> Tuple[str, Dict[str, object]]:
    fid = match_rule_based(user_text, FAQS)
    if fid:
        return get_answer_by_id(fid, FAQS), {"method": "offline_tag_match", "faq_id": fid, "score": 100}

    mr = match_similarity(user_text)
    if not mr:
        return DEFAULT_FALLBACK, {"method": "offline_none", "faq_id": None, "score": 0}

    threshold = st.session_state.get("match_threshold", 55)
    if mr.score < threshold:
        return DEFAULT_FALLBACK, {
            "method": "offline_similarity_below_threshold",
            "faq_id": mr.faq_id,
            "score": mr.score,
            "matched_question": mr.matched_question,
        }

    return get_answer_by_id(mr.faq_id, FAQS), {
        "method": "offline_similarity",
        "faq_id": mr.faq_id,
        "score": mr.score,
        "matched_question": mr.matched_question,
    }


# ----------------------------
# OPENAI-POWERED REPLY (FAQ-grounded)
# ----------------------------
def build_faq_context(faqs: List[Dict[str, object]]) -> str:
    lines = []
    for f in faqs:
        lines.append(f"- FAQ_ID: {f['id']}")
        for q in f["questions"]:
            lines.append(f"  Q: {q}")
        lines.append(f"  A: {f['answer']}")
        lines.append("")
    return "\n".join(lines).strip()

FAQ_CONTEXT = build_faq_context(FAQS)

SYSTEM_INSTRUCTIONS = f"""
You are a helpful school FAQ chatbot for {SCHOOL_NAME} in {SCHOOL_LOCATION}.
You must answer using ONLY the provided FAQ knowledge base.
If asked for contact details, provide exactly:
Phone: {SCHOOL_PHONE}
Email: {SCHOOL_EMAIL}

Do not invent phone numbers, emails, dates, policies, or staff details.
Keep replies short, clear, and friendly.
If the answer is not clearly in the FAQ knowledge base, say you are not sure and suggest contacting the school office.
"""

def get_openai_key() -> Optional[str]:
    # Streamlit Cloud: put OPENAI_API_KEY in Secrets.
    # Local: can be in .streamlit/secrets.toml OR environment.
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None
    return key or os.getenv("OPENAI_API_KEY")

def openai_reply(user_text: str, chat_history: List[Dict[str, str]], model: str) -> Tuple[str, Dict[str, object]]:
    api_key = get_openai_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package not available")

    client = OpenAI(api_key=api_key)

    # Keep chat history small
    trimmed = chat_history[-10:]
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in trimmed])

    prompt = f"""
FAQ KNOWLEDGE BASE:
{FAQ_CONTEXT}

CHAT HISTORY:
{history_text}

USER QUESTION:
{user_text}

Answer now.
"""

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
    )

    answer = (resp.output_text or "").strip() or DEFAULT_FALLBACK
    return answer, {
        "method": "openai_responses_api",
        "model": model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title=f"{SCHOOL_NAME} FAQ Chatbot", page_icon="🤖", layout="centered")

st.title(f"🤖 {SCHOOL_NAME} FAQ Chatbot")
st.caption(f"📍 {SCHOOL_LOCATION} • Ask about admissions, fees, uniform, term dates, reporting time, etc.")

with st.sidebar:
    st.header("⚙️ Settings")
    use_openai = st.toggle("Use OpenAI (online)", value=True)
    st.session_state.match_threshold = st.slider(
        "Offline matching confidence threshold",
        0, 100, 55,
        help="Used only if OpenAI is off or missing.",
    )
    model = st.text_input("OpenAI model", value="gpt-5.2")
    st.write("OpenAI available:", "✅" if OPENAI_AVAILABLE else "❌ (check requirements.txt)")
    st.write("API key detected:", "✅" if bool(get_openai_key()) else "❌ (add to Streamlit Secrets)")
    st.divider()
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hi! I’m the {SCHOOL_NAME} FAQ bot. Ask me something like: *How do I apply?*"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Type your question…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if use_openai:
        try:
            answer, debug = openai_reply(prompt, st.session_state.messages, model=model)
        except Exception as e:
            answer, debug = offline_reply(prompt)
            debug["openai_error"] = str(e)[:200]
    else:
        answer, debug = offline_reply(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Debug (how the answer was made)"):
            st.json(debug)

with st.expander("Admin: View / edit FAQ dataset"):
    st.json(FAQS)