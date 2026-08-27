import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import streamlit as st

# The RAG script has no .py extension, so spec_from_file_location can't infer
# a loader from the suffix (it returns None) — pass a SourceFileLoader
# explicitly. Loading it this way sets its __name__ to the module name below
# (not "__main__"), so its own CLI input loop never fires here.
MODULE_PATH = Path(__file__).parent / "FDA warning Letter RAG"
MODULE_NAME = "fda_warning_letter_rag"


@st.cache_resource(show_spinner="Connecting to Pinecone and loading models (one-time per server start; letters are already ingested)...")
def load_generate_answer():
    loader = SourceFileLoader(MODULE_NAME, str(MODULE_PATH))
    spec = importlib.util.spec_from_loader(MODULE_NAME, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module.generate_answer


generate_answer = load_generate_answer()

st.set_page_config(page_title="FDA Warning Letter RAG", page_icon="⚠️")
st.title("FDA Warning Letter Chatbot")
st.caption("Ask questions about FDA Warning Letters. Answers are grounded only in the ingested letter corpus.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask about an FDA warning letter..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(user_question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
