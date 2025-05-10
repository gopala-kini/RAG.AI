import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import os
from langchain.schema import Document #Schema created in backend
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

#API Configuration
genai.configure(api_key=os.getenv("GOOGLE-GEMINI-API-KEY"))
gemini_model=genai.GenerativeModel("gemini-2.0-flash")
#Cache the HF Embeddings to avoid slow relaod of the embeddings
@st.cache_resource(show_spinner='Loading Embedding Model...')

def embeddings():
    return(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
embedding_model=embeddings()

#user interface
st.header("RAG Assistant:orange[HF Embeddings +Gemini LLM]")
st.subheader("Your AI Doc Assistant")
uploaded_file=st.file_uploader(label='Upload the PDF document',type=['pdf'])

if uploaded_file:
    raw_text=""
    pdf=PdfReader(uploaded_file)
    for i, page in enumerate(pdf.pages):
        text=page.extract_text()
        if text:
            raw_text+=text
    if raw_text.strip():
        document=Document(page_content=raw_text)
        #using chartextsplitter we will create chunks and pass it into the model
        splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks=splitter.split_documents([document])
        
    #Store the chunks into Faiss vectorDB
    chunk_pieces=[chunk.page_content for chunk in chunks]
    vectordb=FAISS.from_texts(chunk_pieces,embedding_model)#convert them into embeddings
    retriever=vectordb.as_retriever() # Retrieve the VECTORS...
    
    st.success("Embeddings are Generated. Ask your Question!!")
    #User Q and A
    user_input=st.text_input(label="Enter your Question")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.spinner("Analyzing the document..."):
            relevant_docs=retriever.get_relevant_documents(user_input)
            context="\n\n".join([doc.page_content for doc in relevant_docs])
            
            prompt=f'''You are an expert assistant. Use the context below to answer
            the query. if unsure or informtion not available in the doc, pass the message-
            "Information is not available." Look into other sources
            context:{context},
            query:{user_input},
            Answer:'''
            response=gemini_model.generate_content(prompt)
            st.markdown("Answer: ")
            st.write(response.text)
else:
    st.warning("No Text could be extracted from PDF. Please upload a readable PDF")            
            