import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings,SentenceTransformerEmbeddings,HuggingFaceBgeEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQAWithSourcesChain,RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores import FAISS , Qdrant
from qdrant_client import QdrantClient
from langchain.document_loaders import PyPDFLoader

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
OPENAI_API_KEY=st.sidebar.text_input("Enter Your API KEY HERE:",type="password")
st.sidebar.markdown("Get your API key from [here](https://platform.openai.com/account/api-keys)")
st.sidebar.divider()
st.sidebar.subheader("Model Selection")
llm_model_options=['gpt-3.5-turbo', 'gpt-3.5-turbo-16k','gpt-4']
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
        - Currently only supports Structured PDFs only. 
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

if "conversation" not in st.session_state:
    st.session_state.conversation = None

st.markdown(f"""## AI-Assisted Q&A from Document 📑""",unsafe_allow_html=True)
st.write("_A tool built for AI-Powered Research Assistance or Querying Documents for Quick Information Retrieval_")

# def speak(answer):
#     engine=pyttsx3.init()
#     voices=engine.getProperty('voices')
#     rate=engine.getProperty('rate')
#     engine.setProperty('rate', int(rate/1.20))
#     flag=True
#     for voice in voices:
#         if 'english' in voice.languages and 'indian' in voice.name.lower():
#             engine.setProperty('voice', voice.id)
#             flag = False
#             break

#     if flag:
#         engine.setProperty('voice', voices[0].id)
#     engine.say(answer)
#     engine.runAndWait()
    
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        try:
            pdf_reader=PdfReader(pdf)
        except (PdfReader.PdfReadError, PyPDF2.utils.PdfReadError) as e:
            print(f"Failed to read {pdf}: {e}")
            continue
        for page in pdf_reader.pages:
            page_text=page.extract_text()
            if page_text:
                text+=page_text
            else:
                print(f"Failed to extract text from a page in {pdf}")
    return text

def get_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(separators='\n',chunk_size=1000,chunk_overlap=200,length_function=len)
    chunks=text_splitter.split_documents(text)
    return chunks

# model_name="BAAI/bge-large-en"
# model_kwargs={'device':'cpu'}
# encode_kwargs={'normalize_embeddings':True}
# embeddings=HuggingFaceBgeEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)

def vectorstores(chunks):
    model_name="BAAI/bge-large-en"
    model_kwargs={'device':'cpu'}
    encode_kwargs={'normalize_embeddings':True}
    embeddings=HuggingFaceBgeEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)
    url="http://localhost:6333"
    collection_name="gpt_db1"
    vector=Qdrant.from_documents(documents=chunks,url=url,embedding=embeddings,prefer_grpc=False,collection_name=collection_name)
    return vector

@st.cache(ttl=24*3600)
def get_conversation_chain(vector):
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=temperature_input,model=model_select,openai_api_key=OPENAI_API_KEY),chain_type=chain_select,
    retriever=vector.as_retriever(),
    get_chat_history=lambda h:h,
    memory=memory)
    return conversation_chain

user_uploads=st.file_uploader("Upload Your File Here",accept_multiple_files=True)
if user_uploads is not None:
    if st.button("Upload"):
        with st.spinner("Processing"):
            raw_text=get_pdf_text(user_uploads)
            text_chunks=get_chunks(raw_text)
            vector_stores=vectorstores(text_chunks)
            st.session_state.conversation = get_conversation_chain(vector_stores)

if 'doc_messages' not in st.session_state or clear_history:
    st.session_state['doc_messages'] = [{"role": "assistant", "content": "Query your documents"}]
    st.session_state['chat_history'] = []  # Initialize chat_history as an empty list

for message in st.session_state['doc_messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])
        
#voice_search_button = st.button("🎤", key="voice_search_button")
#current_mode = "text_input"

# if voice_search_button:
#     current_mode = "voice_search"
    
# if current_mode == "text_input":
user_query = st.chat_input("Enter your query here")

# elif current_mode == "voice_search":
#     user_query = get_audio_input()
    
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
                {
                    "role": "user",
                    "content": user_query
                }
            ]
            result = st.session_state.conversation({
                "question": user_query, 
                "chat_history": st.session_state['chat_history']})
            response = result["answer"]
            st.session_state['chat_history'].append({
                "role": "assistant",
                "content": response
            })
        else:
            response = "Please upload a document first to initialize the conversation chain."
        with st.chat_message("assistant"):
            st.write(response)
            #speak(response)
        st.session_state['doc_messages'].append({"role": "assistant", "content": response})