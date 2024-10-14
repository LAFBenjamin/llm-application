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
def main():
    pdf_icon_url = "https://img.icons8.com/fluency/48/000000/pdf.png"
    load_dotenv()
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
    )
   
    pdf_files = ['doc_rh.pdf', 'doc_formation.pdf']
    # Liste pour stocker les documents chargés
    all_documents = []
    # Charger chaque fichier PDF un par un
    for pdf_file in pdf_files:
        documents = PyPDFLoader(pdf_file).load()
        all_documents.extend(text_splitter.split_documents(documents))

    db = FAISS.from_documents(all_documents, OpenAIEmbeddings(api_key=os.getenv("OPENIA_API_KEY")))

    custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. 
    If you cannot find the answer in the document provided, answer that you can only answer HR questions.
    you will provide a personalized response based on the user context.
    at the end of the reply you ask a question to continue the discussion 
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    you will provide a personalized response based on the user context and you can only answer HR questions.
    """
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,output_key="answer")
    llm = ChatOpenAI(temperature=0.5, openai_api_key=os.getenv("OPENIA_API_KEY"))
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory,
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        return_source_documents=True,
    )

    st.set_page_config(page_title="MyAssistant", page_icon="")
    st.header('Welcome to your HR AI assistant') 

    user_query = st.text_input("**How can I help you?**",
    placeholder="Ask me anything!")
 
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
            print(response["source_documents"])
            unique_sources = set()
            for idx, doc in enumerate(response["source_documents"]):
                source = doc.metadata.get('source', 'N/C')
                if source not in unique_sources:
                    st.markdown(f"### Document {idx+1}")
                    st.markdown(f"![PDF Icon]({pdf_icon_url}) **Source**: {source}")
                    unique_sources.add(source)

    if st.sidebar.button("Reset chat history"):
        st.session_state.messages = []
        st.chat_message()




if __name__ == "__main__":
    main()