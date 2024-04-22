# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import openai
import logging

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

# Load environment variables from .env file
load_dotenv(dotenv_path='.env')

# Accessing the API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
chroma_server_host = os.getenv("CHROMA_SERVER_HOST")
chroma_server_http_port = 8000
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Setup HTTP client for Chroma
db = chromadb.HttpClient(host=chroma_server_host, port=chroma_server_http_port)

# -------------------------------------------- #

import streamlit as st

st.write("GovChat")
# -------------------------------------------- #

import streamlit as st

# Dropdown to select the department
collection_options = {
    'Philippine Health Insurance Corporation': 'PhilHealth_raw_data_collection',
    'Department of Trade and Industry': 'DTI_raw_data_collection',
    'Land Transportation Office': 'LTO_raw_data_collection',
    'Bureau of Internal Revenue': 'BIR_raw_data_collection',
    'Department of Foreign Affairs': 'DFA_raw_data_collection',
    'Social Security System': 'SSS_raw_data_collection',
    'Home Development Mutual Fund (Pag-IBIG)': 'PAGIBIG_raw_data_collection',
    'Department of Education': 'DepEd_raw_data_collection',
    'Philippine Statistics Authority': 'PSA_raw_data_collection'
}

# Department Descriptions
department_details = {
    'Philippine Health Insurance Corporation': 'Responsible for providing health insurance coverage and ensuring affordable, acceptable, available, and accessible health care services for all citizens of the Philippines.',
    'Department of Trade and Industry': 'Works to create a conducive business environment, boost industry growth, and develop consumer protection policies.',
    'Land Transportation Office': 'In charge of all land transportation management, including vehicle registration and driver licensing.',
    'Bureau of Internal Revenue': 'Responsible for collecting taxes, enforcing tax laws, and ensuring tax compliance.',
    'Department of Foreign Affairs': 'Manages the Philippines\' international diplomatic relations, consular services, and passport issuance.',
    'Social Security System': 'Provides social insurance, retirement benefits, and loans for the working population.',
    'Home Development Mutual Fund (Pag-IBIG)': 'Provides affordable housing finance and savings programs for Filipino workers.',
    'Department of Education': 'Oversees the Philippine education system, including elementary, secondary, and non-formal education.',
    'Philippine Statistics Authority': 'Responsible for collecting, analyzing, and publishing statistical information on economic, social, and demographic topics.'
}

# Department Sample Questions
department_questions = {
    'Philippine Health Insurance Corporation': ['How to apply for PhilHealth?', 'What are the benefits of PhilHealth?', 'How to update PhilHealth member information?'],
    'Department of Trade and Industry': ['How to register a business in the Philippines?', 'What are the consumer rights for returns and refunds?', 'How to apply for a trade license?'],
    'Land Transportation Office': ['How to renew a driver\'s license?', 'What are the requirements for vehicle registration?', 'How to report a traffic violation?'],
    'Bureau of Internal Revenue': ['How to file an annual tax return?', 'What are the penalties for late tax payment?', 'How to get a tax clearance certificate?'],
    'Department of Foreign Affairs': ['How to renew a Philippine passport?', 'How to authenticate documents for overseas use?', 'What are the requirements for a diplomatic visa?'],
    'Social Security System': ['How to apply for an SSS loan?', 'What are the retirement benefits under SSS?', 'How to check SSS contribution records?'],
    'Home Development Mutual Fund (Pag-IBIG)': ['How to apply for a Pag-IBIG housing loan?', 'What are the benefits of Pag-IBIG membership?', 'How to withdraw Pag-IBIG savings?'],
    'Department of Education': ['How to apply for the SHS Voucher Program?', 'What are the requirements for public school enrollment?', 'How does the K-12 program work?'],
    'Philippine Statistics Authority': ['How to get a copy of a birth certificate?', 'What are the latest population statistics?', 'How to access agricultural data?']
}

logging.basicConfig(level=logging.INFO)

# Streamlit app design
st.sidebar.title('Department Information Portal')

# Initialize session state for tracking department selection changes
if 'last_selected_department' not in st.session_state:
    st.session_state.last_selected_department = None

# User selects the department from the sidebar
collection_name = st.sidebar.selectbox('Choose a department:', list(collection_options.keys()))

# Function to handle department selection changes
def on_department_change():
    if st.session_state.last_selected_department != collection_name:
        st.session_state.last_selected_department = collection_name
        # Reset the chat messages when the department changes
        st.session_state.messages = [
            {"role": "assistant", "content": "Please begin by sharing your question or concern about {collection_name}."}
        ]
        st.experimental_rerun()  # This will reload the app

# Run the change handler function when the collection_name changes
on_department_change()

# Display the details of the selected department in the sidebar
st.sidebar.subheader(f"{collection_name} Information:")
st.sidebar.write(department_details[collection_name])

# Display sample questions for the selected department in the sidebar
st.sidebar.subheader(f"Sample Questions for {collection_name}:")
for question in department_questions[collection_name]:
    st.sidebar.write("- " + question)

# Retrieve the selected collection
selected_collection = collection_options[collection_name]
chroma_collection = db.get_collection(selected_collection)

# Log the loaded collection
logging.info(f"Loaded collection '{collection_name}' with ID {selected_collection}")

# Create vector store and index
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

# ---------------------------------------------- #

chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Please begin by sharing your question or concern about Indigenous peoples' rights."}
    ]

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

# -------------------------------------------- #