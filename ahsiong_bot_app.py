import streamlit as st
from rag_chatbot import process_user_query
from rag_chatbot import AllergyChecker

checker = AllergyChecker()
checker.insert_data_if_needed()

st.set_page_config(page_title="Ah Siong Bot", page_icon="🛒", layout="centered")
st.title("🧑‍🍳 Ah Siong Bot")
st.caption("Ask me about ingredients, allergies, or food products!")

if "history" not in st.session_state:
    st.session_state.history = []
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

if "pending_input" in st.session_state:
    user_input = st.session_state.pending_input
    del st.session_state["pending_input"]
    response = process_user_query(user_input)
    st.session_state.history.insert(0, (user_input, response))
    st.rerun()

user_input = st.text_input("💬 What would you like to ask?", key="chat_input", on_change=lambda: st.session_state.update(pending_input=st.session_state.chat_input))

# Display the chat history (newest at bottom)
for user_msg, bot_msg in st.session_state.history:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(f"**You:** {user_msg}")

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"**Ah Siong Bot:** {bot_msg}")