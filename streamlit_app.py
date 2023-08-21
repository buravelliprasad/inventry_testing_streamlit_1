import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
import pandas as pd

user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="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",
    type="password")
st.image("socialai.jpg")
# Path to the CSV file
csv_file_path = r"dealer_1_inventry.csv"

# Display the introductory information
st.info("Introducing Engage.ai, your cutting-edge partner in streamlining dealership and customer-related operations. At Engage, we specialize in harnessing the power of automation to revolutionize the way dealerships and customers interact. Our advanced solutions seamlessly handle tasks, from managing inventory and customer inquiries to optimizing sales processes, all while enhancing customer satisfaction. Discover a new era of efficiency and convenience with us as your trusted automation ally. [engane.ai](https://funnelai.com/). For this demo application, we will use the Inventory Dataset. Please explore it [here](https://github.com/buravelliprasad/inventry_testing_streamlit_1/blob/main/dealer_1_inventry.csv) to get a sense for what questions you can ask.")

# Load the CSV data using pandas
data = pd.read_csv(csv_file_path)

# Convert the data to text format
texts = data.astype(str).agg(" ".join, axis=1).tolist()

# Set up OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)

# Create vectorstore using FAISS from text data
vectorstore = FAISS.from_texts(texts, embeddings)

# Create the conversational retrieval chain
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo-16k'),
    retriever=vectorstore.as_retriever())

def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask me anything 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

# Container for the chat history
response_container = st.container()
# Container for the user's text input
container = st.container()
with container:
    
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Type your question here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
