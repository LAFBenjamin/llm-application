# HR AI Assistant - Example Application using LangChain

This application is an AI assistant specialized in Human Resources (HR), using a language model (LLM) to answer questions based on provided PDF documents. The application is built with **LangChain**, **Streamlit**, and the OpenAI API.

## Features

- **PDF Document Loading**: The application loads and splits PDF documents into chunks for more efficient searching.
- **Conversational Search**: You can ask questions related to the PDF documents, and the AI will respond based on these documents.
- **Conversational Memory**: The application remembers the conversation history to improve contextual answers.
- **Personalization**: The AI assistant is configured to only answer HR-related questions.
- **User Interface with Streamlit**: The application provides a simple, user-friendly web interface to interact with the AI.

## Requirements

Before running the application, ensure you have the necessary dependencies installed:

- **Python 3.7+**
- **LangChain**
- **OpenAI** (for OpenAI API)
- **Streamlit**
- **FAISS** (for vector search)
- **PyPDFLoader** (for loading PDF documents)
- **dotenv** (for environment variable management)

## Installation

1. Clone this repository or copy the source code into your local environment.

2. Install the required dependencies by running:

   ```bash
   pip install langchain openai streamlit faiss-cpu pypdf python-dotenv