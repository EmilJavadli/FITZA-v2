
import streamlit as st
import os
import faiss
from pdf_parser import extract_tables_from_pdf
from embed_store import build_faiss_index
from rag_pipeline import retrieve_context, generate_answer


PDF_PATH = "data/pizza_ingredients.pdf"
INDEX_PATH = "faiss_index/index.faiss"

st.set_page_config(page_title="🍕 FITZA Health Assistant", layout="centered")
st.title("🍕 FITZA Health Assistant")
st.write("Ask health-related questions about pizzas!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_query = st.chat_input("Ask a health question about pizza...")
if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Load and process data
    df = extract_tables_from_pdf(PDF_PATH)
    json_dict = df.to_dict(orient="records")

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = build_faiss_index(json_dict)
        faiss.write_index(index, INDEX_PATH)

    # Retrieve context and generate answer
    context = retrieve_context(user_query, index, json_dict)
    answer = generate_answer(context, user_query)

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)


#python -m streamlit run app.py