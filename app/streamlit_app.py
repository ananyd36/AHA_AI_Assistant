import streamlit as st
from sidebar import display_sidebar
from chat_interface import display_chat_interface

st.title("AHA! Chatbot")
st.markdown("Ask questions about the Edge AI curriculum and get instant answers based on the uploaded documents.")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"

# Display the sidebar
display_sidebar()

# Clear chat button
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("Clear", type="secondary"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

# Display the chat interface
display_chat_interface()