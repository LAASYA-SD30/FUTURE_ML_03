import streamlit as st
import ft3  # your existing chatbot logic (make sure ft3.py has functions you can reuse)

st.set_page_config(page_title="Customer Support Chatbot", page_icon="🤖")

st.title("🤖 Customer Support Chatbot")

# Keep conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me something..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Call your chatbot logic here
    # Right now, just using fallback response
    response = ft3.get_bot_response(prompt) if hasattr(ft3, "get_bot_response") else "Bot: Sorry, I don’t understand yet."

    # Add bot response
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
