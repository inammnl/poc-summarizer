import os, tempfile
import base64
import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from streamlit_supabase_auth import login_form, logout_button
from dotenv import load_dotenv


def embed_viewer(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

load_dotenv()

# Streamlit app
st.title('Eiffel Doc Summarizer')

session = login_form(
    url=os.environ.get('SUPABASE_URL'),
    apiKey=os.environ.get('SUPABASE_KEY'),
    # providers=[""]
)

if session:
    
    # Get document input
    source_doc = st.file_uploader("Upload Source Document", type="pdf")

    #what summarizer chain to use
    sumchain = st.selectbox('Select a chain', ('map_reduce', 'stuff', 'refine'))
    print('you selected ', sumchain)

    #length of the summary
    sumlen = st.selectbox('Summary size', ('100', '500', '1000'))
    print('length selected ', sumlen, ' words')

    with st.sidebar:
        st.write(f"Welcome {session['user']['email']}")
        logout_button()

    # Check if the 'Summarize' button is clicked

    if st.button("Summarize"):
        try:
            # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load_and_split()
            embed_viewer(tmp_file.name)
            os.remove(tmp_file.name)
            
            # Create embeddings for the pages and insert into Chroma database
            embeddings=OpenAIEmbeddings()
            vectordb = Chroma.from_documents(pages, embeddings)

            # Initialize the OpenAI module, load and run the summarize chain
            llm=OpenAI(temperature=0)
            #options: map_reduce, stuff, refine
            chain = load_summarize_chain(llm, chain_type=sumchain)
            search = vectordb.similarity_search(" ")
            summary = chain.run(input_documents=search, question="Write a summary within {sumlen} words.")
            
            st.write(summary)
        except Exception as e:
            st.write(f"An error occurred: {e}")