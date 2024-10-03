from langchain_community.embeddings import OpenAIEmbeddings , HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatAnthropic,ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os ,PyPDF2 , pyttsx3 , pyaudio
from PyPDF2 import PdfReader
from langchain.schema import Document
from io import BytesIO
import speech_recognition as sr
from langchain.retrievers import BM25Retriever , EnsembleRetriever

home_privacy = "We value and respect your privacy. To safeguard your personal details, we utilize the hashed value of your OpenAI API Key, ensuring utmost confidentiality and anonymity. Your API key facilitates AI-driven features during your session and is never retained post-visit. You can confidently fine-tune your research, assured that your information remains protected and private."
st.set_page_config(
    page_title="Document Q&A with AI",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-attachment: fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.subheader("Setup")
OPENAI_API_KEY=st.sidebar.text_input("Enter Your Api Key Here : ",type="password")
st.sidebar.markdown("Get Your Api Key from [here](https://platform.openai.com/account/api-keys)")
st.sidebar.divider()
st.sidebar.subheader("Model Selection")
llm_model_options=['gpt-3.5-turbo', 'gpt-3.5-turbo-16k','gpt-4','gpt-4o-mini']
model_select=st.sidebar.selectbox('Select LLM Model:',llm_model_options,index=2)
chain_type_options=['stuff', 'map_reduce', "refine", "map_rerank"]
chain_select=st.sidebar.selectbox('Select Chain Type:',chain_type_options,index=0)
st.sidebar.markdown("""\n""")
temperature_input = st.sidebar.slider('Set Temperature/Randomness:', min_value=0.0, max_value=1.0, value=0.5)
st.sidebar.markdown("""\n""")
clear_history = st.sidebar.button("Clear conversation history")

with st.sidebar:
    st.divider()
    st.subheader("Limitations:", anchor=False)
    st.info(
        """
        - Currently only supports PDFs, Txt and Urls only. 
        """)
    st.divider()
    
with st.sidebar:
    st.subheader("👨‍💻 Author: **Aman Ratnam**", anchor=False)
    
    st.subheader("🔗 Contact / Connect:", anchor=False)
    st.markdown(
        """
        - [Email](mailto:ratnamaman21@gmail.com)
        - [LinkedIn](https://www.linkedin.com/in/aman-ratnam-119503141/)
        - [Github Profile](https://github.com/Ratnam98)
        """
    )
    st.divider()
    st.write("Made with 🦜️🔗 Langchain and OpenAI LLMs")

def speak(answer):
    engine=pyttsx3.init()
    voices=engine.getProperty('voices')
    rate=engine.getProperty('rate')
    engine.setProperty('rate',int(rate/1.20))
    for i in range(len(voices)):
        if 'Zira' in voices[i].name:
            engine.setProperty('voice',voices[i].id)
    engine.say(answer)
    engine.runAndWait()

recognizer=sr.Recognizer()
def get_audio_input():
    with sr.Microphone as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio=recognizer.listen(source,timeout=5)
        st.write("Processing Your Audio")

    try:
        user_query=recognizer.recognize_google_cloud(audio)
        return user_query
    except Exception as e:
        return "Sorry , Please Try Again...."


if "conversation" not in st.session_state:
    st.session_state.conversation = None

st.markdown(f"""## AI-Assisted Q&A from Document 📑""",unsafe_allow_html=True)
st.write("_A tool built for AI-Powered Research Assistance or Querying Documents for Quick Information Retrieval_")

user_choice=st.radio(
    "Would you like to process a URL or upload PDF/TXT files?",
    ("URL", "PDF/TXT Files")
)

if user_choice == "PDF/TXT Files":
    user_uploads = st.file_uploader("Upload Your Files Here (PDF, TXT)", accept_multiple_files=True)
    url_input = None
elif user_choice == "URL":
    url_input = st.text_input("Enter a URL Here")
    user_uploads = None

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_txt_text(text_files):
    text=""
    for file in text_files:
        text+= file.read().decode("utf-8")
    return text

def get_url_content(url):
    loader=WebBaseLoader(url)
    documents=loader.load()
    return " ".join([doc.page_content for doc in documents])

def get_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(separators='\n',chunk_size=2000,chunk_overlap=200)
    chunks=text_splitter.split_text(text)
    return chunks

def vectorstores(chunks):
    embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    documents = [Document(page_content=chunk) for chunk in chunks]
    url="http://localhost:6333"
    collection_name="gpt_db2"
    qdrant_client=QdrantClient(url=url)
    qdrant_client.delete_collection(collection_name=collection_name)
    vector=Qdrant.from_texts(texts=chunks,url=url,embedding=embeddings,prefer_grpc=False,collection_name=collection_name)
    return vector

def hybrid_retriever(chunks,vector):
    keyword_retriever=BM25Retriever.from_texts(chunks)
    keyword_retriever.k=3
    vector_stores=vector.as_retriever()
    ensemble_retriever=EnsembleRetriever(retrievers=[vector_stores,keyword_retriever],weights=[0.3, 0.7])
    return ensemble_retriever

def get_conversation_chain(vector):
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    ensemble_retriever = hybrid_retriever(chunks,vector)
    conversation_chain=ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=temperature_input,model=model_select,openai_api_key=OPENAI_API_KEY),chain_type=chain_select,
    retriever=vector.as_retriever(),
    get_chat_history=lambda h:h,
    memory=memory)
    return conversation_chain

# def get_conversation_chain(vector):
#     memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
#     conversation_chain=ConversationalRetrievalChain.from_llm(
#     llm=ChatOpenAI(temperature=temperature_input,model=model_select,openai_api_key=OPENAI_API_KEY),chain_type=chain_select,
#     retriever=vector.as_retriever(),
#     get_chat_history=lambda h:h,
#     memory=memory)
#     return conversation_chain

if 'doc_messages' not in st.session_state or clear_history:
    st.session_state['doc_messages'] = [{"role": "assistant", "content": "Query your documents"}]
    st.session_state['chat_history'] = []  # Initialize chat_history as an empty list

for message in st.session_state['doc_messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

if (user_uploads or url_input) and st.button("Process"):
    with st.spinner("Processing..."):
        raw_text = ""
        if user_choice == "PDF/TXT Files" and user_uploads:
            pdf_docs = [file for file in user_uploads if file.name.endswith('.pdf')]
            text_files = [file for file in user_uploads if file.name.endswith('.txt')]
            if pdf_docs:
                raw_text += get_pdf_text(pdf_docs)
            if text_files:
                raw_text += get_txt_text(text_files)
        elif user_choice == "URL" and url_input:
            raw_text += get_url_content(url_input)

        if raw_text:
            text_chunks = get_chunks(raw_text)
            vector_stores = vectorstores(text_chunks)
            st.session_state.conversation = get_conversation_chain(vector_stores)

        if raw_text:
            text_chunks = get_chunks(raw_text)
            vector_stores = vectorstores(text_chunks)
            ensemble_retriever=hybrid_retriever(text_chunks,vector_stores)
            st.session_state.conversation = get_conversation_chain(ensemble_retriever)
            st.session_state.conversation = get_conversation_chain(vector_stores)

user_query = st.chat_input("Enter your query here")
response = ""
if user_query:        
    if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.session_state['doc_messages'].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Generating response..."):
        if 'conversation' in st.session_state:
            st.session_state['chat_history'] = st.session_state.get('chat_history', []) + [
                {"role": "user", "content": user_query}
            ]
            result = st.session_state.conversation({"question": user_query, "chat_history": st.session_state['chat_history']})
            response = result["answer"]
            st.session_state['chat_history'].append({"role": "assistant", "content": response})
            #speak(response)
        else:
            response = "Please upload a document or enter a URL first to initialize the conversation chain."
        with st.chat_message("assistant"):
            st.write(response)
            #speak(response)
        st.session_state['doc_messages'].append({"role": "assistant", "content": response})