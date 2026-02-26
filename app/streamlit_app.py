import streamlit as st
from sidebar import display_sidebar
from chat_interface import display_chat_interface

st.title("AHA Curriculum Assistant")
st.markdown("Ask questions about the AHA curriculum and get answers based on the uploaded documents!")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "model" not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo"  # Default model, can be changed based on user selection in the future

# Display the sidebar
display_sidebar()

# Display the chat interface
display_chat_interface()