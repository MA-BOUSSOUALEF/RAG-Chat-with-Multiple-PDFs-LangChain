
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512},task="text-generation")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, response_container):
    """Gère l'entrée utilisateur et affiche la réponse complète depuis la fin jusqu'à 'Helpful Answer:'."""
    response = st.session_state.conversation({'question': user_question})
    
    if response and 'chat_history' in response:
        last_bot_message = response['chat_history'][-1].content.strip()

        # Trouver la dernière occurrence de "Helpful Answer:"
        helpful_answer_index = last_bot_message.lower().rfind("helpful answer:")
        
        if helpful_answer_index != -1:
            extracted_answer = last_bot_message[helpful_answer_index:].strip()  # Affiche depuis "Helpful Answer:" jusqu'à la fin
        else:
            extracted_answer = last_bot_message  # Si non trouvé, afficher tout le message
        
        # Effacer et afficher uniquement la nouvelle réponse
        response_container.empty()
        with response_container:
            st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", extracted_answer), unsafe_allow_html=True)


def main():
    
    """Interface utilisateur principale avec Streamlit."""
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    load_dotenv()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "processed" not in st.session_state:
        st.session_state.processed = False

    st.header("Chat with PDFs 📚")

    # 🔹 Conteneur pour afficher une seule réponse à la fois
    response_container = st.empty()

    user_question = st.text_input("Ask a question about your documents:", disabled=not st.session_state.processed)
    
    if user_question and st.session_state.processed:
        handle_userinput(user_question, response_container)  # ✅ Correction : on passe bien response_container

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        pdf_url = st.text_input("Or enter a PDF URL")

        if st.button("Process"):
            with st.spinner("Processing..."):
                pdf_paths = []

                if pdf_docs:
                    pdf_paths.extend(pdf_docs)
                
                if pdf_url:
                    downloaded_pdf = download_pdf(pdf_url)
                    if downloaded_pdf:
                        pdf_paths.append(downloaded_pdf)

                if not pdf_paths:
                    st.warning("⚠️ Please upload a PDF or enter a valid PDF URL.")
                else:
                    raw_text = get_pdf_text(pdf_paths)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.processed = True
                    st.success("✅ Processing complete! You can now ask questions.")
                    st.experimental_rerun()  # 🔄 Rafraîchit pour activer le champ de saisie


if __name__ == '__main__':
    main()

