from embed_store import get_embeddings
import google.generativeai as genai
import os
# from dotenv import load_dotenv
import streamlit as st


def retrieve_context(query, index, chunks, k=3):
    query_embedding = get_embeddings([f"Represent this sentence for retrieval: {query}"])
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

def build_dynamic_prompt(context_records, user_query):
    context_text = "\n".join([f"- {record}" for record in context_records])
    
    prompt = (
        "You are a nutrition expert. Based on the following pizza ingredient data, provide a health recommendation.\n"
        "Consider factors such as saturated fat, processed meats, vegetable content, and overall nutritional balance.\n\n"
        f"Pizza entries:\n{context_text}\n\n"
        f"Question: {user_query}\n\n"
        "Instructions:\n"
        "- Provide a concise and clear health recommendation.\n"
        "- Assign a health grade from 0 to 10 (10 = healthiest, 0 = least healthy).\n"
        "- If the data is insufficient, respond only with: 'Insufficient information to provide a recommendation.'\n"
    )

    return prompt

def generate_answer(context_records, query):
    # load_dotenv()
    # gemini_api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    llm = genai.GenerativeModel('gemini-2.0-flash')
    prompt = build_dynamic_prompt(context_records, query)
    response = llm.generate_content(prompt)
    return response.text
