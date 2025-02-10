This project allows users to interact with multiple PDF documents through a chatbot. The app, built with Streamlit, enables users to upload PDF files, which are then converted into text, split into chunks, and indexed using FAISS for fast retrieval.

A HuggingFace and LangChain based language model is used to generate embeddings and answer user queries based on the document content. The app maintains a conversation history for dynamic and continuous interactions, providing contextual responses from multiple documents.