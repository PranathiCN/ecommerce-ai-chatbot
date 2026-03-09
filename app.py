import streamlit as st
import torch
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="E-commerce AI Support Chatbot",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 E-commerce AI Customer Support Chatbot")


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("About")

    st.write(
        "This AI chatbot answers e-commerce customer support questions "
        "using a Retrieval-Augmented Generation (RAG) pipeline."
    )

    st.write("### Tech Stack")

    st.write("- Streamlit")
    st.write("- FAISS Vector Database")
    st.write("- Sentence Transformers")
    st.write("- FLAN-T5 Language Model")


# -----------------------------
# Load LLM
# -----------------------------
@st.cache_resource
def load_model():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,
        device=device
    )

    return generator


generator = load_model()


# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():

    with open("data/ecommerce_data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    return text


text = load_data()


# -----------------------------
# Text Splitting
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

docs = splitter.create_documents([text])


# -----------------------------
# Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Vector Database (Persistent)
# -----------------------------
@st.cache_resource
def create_vectorstore():

    if os.path.exists("faiss_index"):

        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index")

    return vectorstore


vectorstore = create_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# -----------------------------
# Chat Memory
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Display Previous Chat
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User Input
# -----------------------------
query = st.chat_input("Ask a question about orders, payments, returns, etc.")


# -----------------------------
# Generate Response
# -----------------------------
if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            retrieved_docs = retriever.invoke(query)

            context = "\n".join(
                [doc.page_content for doc in retrieved_docs]
            )

            prompt = f"""
You are a helpful AI customer support assistant for an e-commerce store.

Use ONLY the information from the context below.

If the answer is not available, say:
"I don't have that information in the knowledge base."

Context:
{context}

Customer Question:
{query}

Answer clearly:
"""

            result = generator(prompt)

            answer = result[0]["generated_text"]

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )