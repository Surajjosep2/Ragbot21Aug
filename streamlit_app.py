import streamlit as st
import os
import tempfile

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
#from langchain_community.vectorstores import AstraDB
from langchain_astradb import AstraDBVectorStore
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_file, vector_store):
    if uploaded_file is not None:
        
        # Write to temporary file
        temp_dir = tempfile.TemporaryDirectory()
        file = uploaded_file
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, 'wb') as f:
            f.write(file.getvalue())

        # Load the PDF
        docs = []
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

        # Create the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap  = 100
        )

        # Vectorize the PDF and load it into the Astra DB Vector Store
        pages = text_splitter.split_documents(docs)
        vector_store.add_documents(pages)  
        st.info(f"{len(pages)} pages loaded.")

# Cache prompt for future runs
@st.cache_data()
def load_prompt():
    template = """You are an AI assistant skilled in generating RFP templates tailored to specific business needs. You have good understanding of major listed companies in India. Also, you have good grasp on professional services trend in Indian market. You also have knowledge on a collection of marketing and business advisory books. Also, ask more questions until their is detailed understanding of user business and his location. Please use your knowlege to create this RFP template. Iterate and keep asking questions until if you are able to match some of services of professional services firm under project scope. But don't mention the names Big4, EY, Deloitte, KPMG, PwC or their related and associated brand names. Also keep asking question until some of your business consulting frameworks are used. You will structure your responses using the following sections:

**1. Introduction:** 
* Provide a concise overview of the project or initiative prompting the RFP.
* Briefly describe the organization issuing the RFP.

**2. Project Scope:** 
* Clearly outline the specific goals, objectives, and deliverables expected from the project.
* Include any key milestones or timelines that are relevant.

**3. Vendor Requirements:** 
* Specify the qualifications, experience, and certifications desired from potential vendors.
* Include any technical or industry-specific requirements.

**4. Proposal Guidelines:** 
* Clearly state the format and structure expected for vendor proposals.
* Define the evaluation criteria that will be used to assess proposals.

**5. Submission Details:**
* Provide the deadline for proposal submission.
* Specify the contact person or department for inquiries.

**6. Potential Big 4 Services:**
* Provide the details of some of services of EY, Deloitte, KPMG and PwC that can be consumed. 

**7. Suggested Business Consulting Frameworks:**
* Suggest some business consulting frameworks that can be used. How it can be related to various segments of user's business. Optionally quote some examples from your knowledge.

Randomly provide three Business names, and their related Contact, Address, Email and Phone details. Also, ask follow up question to confirm if business strategy framework applied has some clarity to the user. 

**User Query:**
{question}

**RFP Template:**


CONTEXT:
{context}

QUESTION:
{question}

YOUR ANSWER:"""
    return ChatPromptTemplate.from_messages([("system", template)])
prompt = load_prompt()

# Cache OpenAI Chat Model for future runs
@st.cache_resource()
def load_chat_model():
    return ChatOpenAI(
        temperature=0.3,
        model='gpt-3.5-turbo',
        streaming=True,
        verbose=True
    )
chat_model = load_chat_model()

# Cache the Astra DB Vector Store for future runs
@st.cache_resource(show_spinner='Connecting to Astra')
def load_vector_store():
    # Connect to the Vector Store
    vector_store = AstraDBVectorStore(
        embedding=OpenAIEmbeddings(),
        collection_name="surajlanflowtrialnew",
        api_endpoint=st.secrets['ASTRA_API_ENDPOINT'],
        token=st.secrets['ASTRA_TOKEN']
    )
    return vector_store
vector_store = load_vector_store()

# Cache the Retriever for future runs
@st.cache_resource(show_spinner='Getting retriever')
def load_retriever():
    # Get the retriever for the Chat Model
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
    return retriever
retriever = load_retriever()

# Start with empty messages, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Draw a title and some markdown


st.title("My Friendly Indian Small Business Advisory App")
st.markdown("""Generative AI based solution to boost your productivity and address your day-to-day business activities!

*** Developed by Suraj J.***
Please provide details  requested below to evaluate your business:- 
* Provide a concise overview of the business or initiatives you are trying to drive. Briefly describe the demographies and history of your organization.
* Clearly outline the specific goals, objectives, and deliverables expected from the project.
* Include any key milestones or timelines that are relevant.
* Specify the qualifications, experience, and certifications desired from potential vendors.
* Include any technical or industry-specific requirements.
* Clearly state the format and structure expected for vendor proposals.
* Provide a small brief about your timeline requirements.
""")

# Include the upload form for new data to be Vectorized
with st.sidebar:
    with st.form('upload'):
        uploaded_file = st.file_uploader('Upload a document for additional context', type=['pdf'])
        submitted = st.form_submit_button('Save to Astra DB')
        if submitted:
            vectorize_text(uploaded_file, vector_store)

# Draw all messages, both user and bot so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Draw the chat input box
if question := st.chat_input("What's up?"):
    
    # Store the user's question in a session object for redrawing next time
    st.session_state.messages.append({"role": "human", "content": question})

    # Draw the user's question
    with st.chat_message('human'):
        st.markdown(question)

    # UI placeholder to start filling with agent response
    with st.chat_message('assistant'):
        response_placeholder = st.empty()

    # Generate the answer by calling OpenAI's Chat Model
    inputs = RunnableMap({
        'context': lambda x: retriever.get_relevant_documents(x['question']),
        'question': lambda x: x['question']
    })
    chain = inputs | prompt | chat_model
    response = chain.invoke({'question': question}, config={'callbacks': [StreamHandler(response_placeholder)]})
    answer = response.content

    # Store the bot's answer in a session object for redrawing next time
    st.session_state.messages.append({"role": "ai", "content": answer})

    # Write the final answer without the cursor
    response_placeholder.markdown(answer)
