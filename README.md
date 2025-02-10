# PDF Chatbot with Streamlit and LangChain

This project allows users to interact with multiple PDF documents via a chatbot. The application leverages **Streamlit** for the user interface, **FAISS** for efficient document retrieval, and **HuggingFace** models for generating embeddings and answering user queries.

## Features

- Upload and process multiple PDF files.
- Extract text from PDF files and split them into chunks.
- Use **FAISS** for indexing and efficient retrieval of document content.
- Use **HuggingFace** and **LangChain** for question-answering based on the document text.
- Maintain conversation history for dynamic interactions.

## Technologies Used

- **Streamlit**: For building the interactive web interface.
- **LangChain**: For document processing, embedding generation, and conversation management.
- **FAISS**: For indexing and searching document content efficiently.
- **HuggingFace**: For generating text embeddings with pre-trained models (e.g., `hkunlp/instructor-xl`).
- **PyPDF2**: For extracting text from PDF files.
- **Python-dotenv**: For managing environment variables.