import streamlit as st
from api_utils import get_api_response

def display_chat_interface():
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Query:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            response = get_api_response(prompt, st.session_state.session_id, st.session_state.model)
            
            if response:
                st.session_state.session_id = response.get('session_id')
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                
                with st.chat_message("assistant"):
                    st.markdown(response['answer'])
                    
                    sources = response.get('sources', [])
                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                st.code(source)
            else:
                st.error("Failed to get a response from the API. Please try again.")