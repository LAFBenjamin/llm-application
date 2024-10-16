import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
import streamlit as st
import pandas as pd

from langchain.llms import HuggingFaceHub


PDF_FILES = ['doc_Politiques_Générales et Charte de l’Entreprise.pdf',
            'DOCUMENT 2 : Gestion des Temps de Travail et Congés.pdf',
            'DOCUMENT 3 : Sécurité, Santé et Conditions de Travail.pdf',
            'DOCUMENT 4 : Formation et Développement des Compétences.pdf',
            'DOCUMENT 5 : Avantages Sociaux et Rémunération.pdf',
            'DOCUMENT 6 : Gestion des Conflits et Sanctions.pdf',
            'Procédure de Demande de Congé via l’Outil HoroQuartz.pdf']

PDF_ICON_URL = "https://img.icons8.com/fluency/48/000000/pdf.png"
OPENAI_API_KEY = os.getenv("OPENIA_API_KEY")

def load_and_split_documents(pdf_files, text_splitter):
    all_documents = []
    for pdf_file in pdf_files:
        try:
            documents = PyPDFLoader(pdf_file).load()
            all_documents.extend(text_splitter.split_documents(documents))
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier {pdf_file}: {str(e)}")
    return all_documents
def setup_llm_OpenAI():
    return ChatOpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)
def setup_qa_chain(db, memory):
    custom_template = """
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. 
    If you cannot find the answer in the document provided, answer that you can only answer HR questions.
    You will provide a personalized response based on the user context.
    At the end of the reply, ask a question to continue the discussion.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """
    question_prompt = PromptTemplate.from_template(custom_template)
    llm = setup_llm_OpenAI()
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory,
        condense_question_prompt=question_prompt,
        return_source_documents=True,
    )
def display_source_documents(source_documents):
    unique_sources = set()
    for idx, doc in enumerate(source_documents):
        source = doc.metadata.get('source', 'N/C')
        if source not in unique_sources:
            st.markdown(f"### Document {idx + 1}")
            st.markdown(f"![PDF Icon]({PDF_ICON_URL}) **Source**: {source}")
            unique_sources.add(source)




def main():
    load_dotenv()
    st.set_page_config(page_title="MyAssistant", page_icon="")
    st.header('Welcome to your HR AI assistant')

    # Initialisation des objets
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    all_documents = load_and_split_documents(PDF_FILES, text_splitter)
    db = FAISS.from_documents(all_documents, OpenAIEmbeddings(api_key=OPENAI_API_KEY))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    qa_chain = setup_qa_chain(db, memory)

    # Interface utilisateur
    user_query = st.text_input("**How can I help you?**", placeholder="Ask me anything!")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    if "memory" not in st.session_state:
        st.session_state['memory'] = memory

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            response = qa_chain({"question": user_query})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.write(response["answer"])
            display_source_documents(response["source_documents"])

    # Réinitialisation de l'historique
    if st.sidebar.button("Reset chat history"):
        st.session_state.messages = []

if __name__ == "__main__":
    main()