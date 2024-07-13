import os  
import streamlit as st  
import pickle  
import time  
from langchain import OpenAI  
from langchain.chains import RetrievalQAWithSourcesChain  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.document_loaders import UnstructuredURLLoader  
from langchain.embeddings import OpenAIEmbeddings  
from langchain.vectorstores import FAISS  
from dotenv import load_dotenv  
from urllib.parse import urlparse  

# load environment variables from .env (especially openai api key because thats the only thing there HAHAHAHA)
load_dotenv()

# set up streamlit ui
st.set_page_config(page_title="FinScraper")  # configuring the streamlit app with title

# sidebar ui
st.sidebar.title("News Article URLs")  # setting the title of the sidebar to "news article urls"
urls = []  # initializing an empty list named 'urls' to store input urls
for i in range(3):
    url = st.sidebar.text_input(f"enter url {i+1}")  # creating text input fields in the sidebar for urls
    urls.append(url.strip())  # adding each url to the list after removing any leading or trailing whitespace

process_url_clicked = st.sidebar.button("process urls")  # creating a button in the sidebar to trigger url processing
file_path = "faiss_store_openai.pkl"  # defining the file path for storing serialized data

# main content ui
st.title("finscraper")  # setting the title for the main content area of the app
query = st.text_input("ask a question:")  # creating a text input field for user queries

main_placeholder = st.empty()  # creating an empty placeholder for displaying dynamic content in the main area

llm = OpenAI(temperature=0.9, max_tokens=500)  # initializing an instance of the openai class with specific parameters

# function to validate url format
def is_valid_url(url):
    try:
        result = urlparse(url)  # parsing the url to check its validity
        return all([result.scheme, result.netloc])  # checking if the url has both scheme and network location
    except ValueError:
        return False  # returning false if there's a valueerror indicating an invalid url format

# process urls if button clicked
if process_url_clicked:
    # filter out empty and invalid urls
    valid_urls = [url for url in urls if url and is_valid_url(url)]  # filtering out urls that are empty or invalid
    if not valid_urls:
        st.sidebar.error("please enter valid urls.")  # displaying an error message in the sidebar if no valid urls are entered
    else:
        try:
            # load data
            loader = UnstructuredURLLoader(urls=valid_urls)
            main_placeholder.text("loading data from urls...")  # displaying a loading message in the main area
            data = loader.load()

            # split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("splitting text into chunks...")  # displaying a message for text splitting process
            docs = text_splitter.split_documents(data)

            # create embeddings and save to faiss index
            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("building vector store...")  # displaying a message for vector store building process
            time.sleep(2)

            # save the faiss index to a pickle file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

            main_placeholder.text("processing completed successfully!")  # displaying success message in the main area

        except Exception as e:
            st.sidebar.error(f"error processing urls: {e}")  # displaying an error message in the sidebar if an exception occurs

# answer query if provided
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            # display answer and sources
            st.header("answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
    else:
        st.error("faiss index not found. please process urls first.")  # displaying an error message if faiss index is not found
