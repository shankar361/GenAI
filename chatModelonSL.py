from docx import Document as DocxDocument
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
import tempfile
import dotenv
import os


dotenv.load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""
if "query" not in st.session_state:
    st.session_state.query = ""

llm = ChatOpenAI(
    base_url  = "https://api.groq.com/openai/v1",
    api_key   = os.getenv("api_key"),
    model_name= "llama3-70b-8192"
)
def clear_text():
    st.session_state.query=""

def start_chat(qa):
 
    response = qa.invoke("Is this a resume or is resume uploaded, answer in yes or no.")
    ans = response["result"].lower()
    if "yes" not in ans:
         st.error("The candidate's reusme is not uploaded! Please upload resume")
    else:
        st.write("Dear HR, How can I help you today?")
        query = st.text_input("Enter your query",key="1111111111",on_change=clear_text)
        upQuery=query.upper()
        if upQuery == "Q" or upQuery == "QUIT" or upQuery == "END" or upQuery == "BYE" :
            st.write("Good bye..")
        else:
            with st.spinner("Analyzing data.."):
                response = qa.invoke(query)
                if not response["source_documents"]:
                    st.write("The Question is out of context or the correct documents not provided")
                else:
                    st.write(response["result"]) 
                    st.session_state.chat_history +="You:\n"+response["query"]+"\nHRBuddy:\n"+response["result"]+"\n\n"
                    st.text_area("Conversation with HRBuddy", value=st.session_state.chat_history, height=400)


def load_file(file_path):
    docx  = DocxDocument(file_path)
    text  = "\n".join([p.text for p in docx.paragraphs if p.text.strip()])
    return [Document(page_content=text)]

def splitData(pdfData):
   splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
   splittedpdfDoc = splitter.split_documents(pdfData)
   return splittedpdfDoc


def loadAndSplit_pdfFile(files):
  
    loader  = PyMuPDFLoader(files)
    pdfData = loader.load()
    splittedpdfDoc = splitData(pdfData)
    return splittedpdfDoc

def createTempfile(files):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(files.read())
        return tmp_file.name

def  createRetrieverQA(docs,emb):
    vectorStores = FAISS.from_documents(docs,emb)
    retriever = vectorStores.as_retriever()
    qa = RetrievalQA.from_llm(
    llm = llm,
    retriever = retriever,
    return_source_documents=True       
    )
    return qa


emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docs = []
st.title("Hiring Agent")
st.header("******Your Personalized Hiring assistant******")
path = st.file_uploader("Upload Candidate's Resume and supporting docs(*.pdf, *.docx)",
accept_multiple_files = True,
help="Upload one more more files"
)

if path is not None:
    for files in path:
        tmp = createTempfile(files)
        if files.name.endswith(".pdf"):         
            splittedpdfDoc = loadAndSplit_pdfFile(tmp)
            docs.extend(splittedpdfDoc)
        else:
            docs.extend(load_file(tmp))
  
    if len(docs) > 0:
        with st.spinner("Builing dataset, please wait.."):
            qa = createRetrieverQA(docs,emb)    
        start_chat(qa)
