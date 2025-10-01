import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Customer Support Chatbot", layout="wide")
st.title("🤖 Customer Support Chatbot")

# -------- Step 1: Load CSVs --------
@st.cache_data
def load_csvs():
    try:
        df1 = pd.read_csv("C:/Users/laasy/OneDrive/Desktop/future interns/task 3/twcs/twcs.csv")
        st.write(f"Loaded: twcs.csv — rows: {len(df1)}")
    except:
        df1 = pd.DataFrame()

    try:
        df2 = pd.read_csv("C:/Users/laasy/OneDrive/Desktop/future interns/task 3/sample.csv")
        st.write(f"Loaded: sample.csv — rows: {len(df2)}")
    except:
        df2 = pd.DataFrame()

    df = pd.concat([df1, df2], ignore_index=True)

    # Use only 200 rows for demo
    if len(df) > 200:
        st.warning(f"Dataset has {len(df)} rows, using only 200 for demo...")
        df = df.sample(200, random_state=42)

    return df

df = load_csvs()
st.write(f"Final rows used: {len(df)}")

# -------- Step 2: Build FAQ KB --------
def build_kb_from_csv(df):
    if not {"tweet_id","text","inbound","in_response_to_tweet_id"}.issubset(df.columns):
        return pd.DataFrame(), np.zeros((0,384))

    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true","1","yes","y"])
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")

    replies = df[~df["inbound"]].dropna(subset=["in_response_to_tweet_id"])
    reply_map = replies.set_index("in_response_to_tweet_id")["text"].to_dict()

    records = []
    for _, row in df[df["inbound"]].iterrows():
        q = str(row.get("text","")).strip()
        tid = row.get("tweet_id")
        a = reply_map.get(tid)
        if q and a:
            records.append({"q":q,"a":a})

    faq_df = pd.DataFrame(records)

    if not faq_df.empty:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(faq_df["q"].tolist(), normalize_embeddings=True)
    else:
        embeddings = np.zeros((0,384), dtype=np.float32)

    return faq_df, embeddings

faq_df, embeddings = build_kb_from_csv(df)

# -------- Step 2b: Add fallback FAQ --------
if faq_df.empty:
    st.warning("⚠️ No Q→A pairs found, using fallback FAQ.")

    fallback_data = {
        "q": [
            "how to reset my password",
            "where is my order",
            "how to contact support",
            "what is the refund policy"
        ],
        "a": [
            "To reset your password, go to settings → security → reset password.",
            "You can track your order from the 'My Orders' section in your account.",
            "Contact support via email at support@example.com or call 1800-123-456.",
            "Refunds are processed within 5–7 business days after approval."
        ]
    }

    faq_df = pd.DataFrame(fallback_data)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(faq_df["q"].tolist(), normalize_embeddings=True)

st.write(f"✅ Built FAQ KB with {len(faq_df)} Q→A pairs")

# -------- Step 3: Guided Flows --------
def guided_flow(intent):
    if intent == "order_status":
        with st.form("order_status_form"):
            order_id = st.text_input("Order ID", placeholder="e.g., ORD12345")
            email = st.text_input("Email used for the order")
            submitted = st.form_submit_button("Check Status")
            if submitted:
                if order_id and email:
                    st.success(f"Status check for **{order_id}** queued. Updates will be sent to **{email}**.")
                else:
                    st.warning("Please fill in both Order ID and Email.")

    elif intent == "refund":
        with st.form("refund_form"):
            order_id = st.text_input("Order ID")
            reason = st.text_area("Reason for refund")
            submitted = st.form_submit_button("Submit Refund Request")
            if submitted:
                if order_id and reason:
                    st.success(f"Refund request for **{order_id}** received. Our team will contact you within 24–48 hours.")
                else:
                    st.warning("Please fill all fields.")

# -------- Step 4: Simple Intent Detection --------
import re
INTENTS = {
    "order_status": [r"order status","track order","where is my order","delivery status"],
    "refund": [r"refund","return","money back"]
}

def detect_intent(text):
    t = text.lower()
    for intent, patterns in INTENTS.items():
        for p in patterns:
            if re.search(p, t):
                return intent
    return "faq"

# -------- Step 5: Chatbot Response --------
def chatbot_reply(query):
    intent = detect_intent(query)
    if intent in ["order_status","refund"]:
        guided_flow(intent)
        return f"Guided flow for **{intent.replace('_',' ')}** displayed above."

    if faq_df.empty:
        return "❌ No FAQ available."

    q_emb = embedder.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    best_idx = np.argmax(sims)
    if sims[best_idx] < 0.4:
        return "⚠️ Sorry, I don’t know that yet."
    return faq_df.iloc[best_idx]["a"]

# -------- Step 6: Chat Input --------
user_input = st.text_input("Ask me something:")
if user_input:
    response = chatbot_reply(user_input)
    st.write(f"**Bot:** {response}")
