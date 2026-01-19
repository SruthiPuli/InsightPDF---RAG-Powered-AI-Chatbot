import streamlit as st 
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


#--------------------------------Setting Layout------------------------------------#

st.set_page_config(layout='wide')

#--------------------------------CSS Styling-----------------------------------------#

st.markdown("""
<style>
            
    [data-testid="stChatMessage"] {
        max-width: 75%;
    }
            
    p, li, textarea{
        font-weight : 500 !important; 
    }
            
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        margin-left: auto;        
        width: fit-content;
        
    }
            
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color : rgba(240, 242, 246, 0.5);
        padding : 20px;
        width: fit-content;
    }
  

             

    
</style>
""", unsafe_allow_html=True)

#-----------------------------Setting Title--------------------------------------#

st.title("PDF RAG CHATBOT - An AI Document Assistant")

#--------------------------------Loading Embedding Model----------------------------#

@st.cache_resource(show_spinner=False)
def load_embedding_model() :

    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

#------------------------------Getting Groq API Key---------------------------------#

def get_groq_api_key() :

    groq_api_key = None

    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]

    else :
        
        from dotenv import load_dotenv
        load_dotenv()
        groq_api_key = os.getenv('GROQ_API_KEY')
        #os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']

    return groq_api_key

#----------------------------------Loading LLM---------------------------------------------#

@st.cache_resource(show_spinner=False)
def load_llm(api_key) :

    os.environ["GROQ_API_KEY"] = api_key

    llm = ChatGroq(api_key = api_key, model = "llama-3.1-8b-instant", temperature = 0.3)
    return llm

#---------------------------------Reading and Storing the Document---------------------------#
@st.cache_data(show_spinner=False)
def process_pdf(pdf_file, embedding_model) :

    pdf_reader = PdfReader(pdf_file)
    text = ""

    for page in pdf_reader.pages :
        if(page.extract_text()) :
            text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, 
        chunk_overlap = 100,
        separators = ['\n\n', '\n'],
        keep_separator=True,
        length_function = len
    )

    chunks = text_splitter.split_text(text=text)

    knowledge_base = FAISS.from_texts(chunks, embedding_model)

    return knowledge_base, chunks


#--------------------------------Generating Recommended Questions-----------------------------#

def generate_recommended_questions(context) :

    prompt = f"""
        You are a helpful assistant.

        TASK:
        Generate exactly 3 useful questions a user may ask to understand the document.

        CONTEXT:
        {context}

        OUTPUT:
        Return ONLY the numbered questions.
        Do NOT include explanations, instructions, or extra text.
    """


    llm = load_llm(get_groq_api_key())
    response = llm.invoke(prompt)

    return [q.strip() for q in response.content.split('\n') if q.strip()]

#-------------------------Asking a query to the LLM-----------------------------#

def ask_query() :

    with st.chat_message('user') :
        st.write(st.session_state.user_query)
        st.session_state.chat_history.append({'role' : 'user', 'content' : st.session_state.user_query})

    with st.spinner("Generating the Response") :

        docs = st.session_state.knowledge_base.similarity_search(st.session_state.user_query, k = 3)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"""

        You are a helpful assistant.
        Answer the question only using the Context below.
        If the answer is not present in the context, Just say,
        Answer is not present in the document.

        Context : 
        {context}

        Question : 
        {st.session_state.user_query}

        """

        llm = load_llm(get_groq_api_key())
        response = llm.invoke(prompt)

    # with st.chat_message('assistant') :
    #     st.write(response.content)
    st.session_state.chat_history.append({'role' : 'assistant', 'content' : response.content})
    
    st.session_state.user_query = None
    st.rerun()
    
#---------------------------Main method----------------------------#

def main() :

    #--------------------Initializing Session Variables--------------------------------#

    if "chat_history" not in st.session_state :

        st.session_state.chat_history = []
        st.session_state.knowledge_base = None
        st.session_state.chunks = []
        st.session_state.user_query = None
        st.session_state.file = None 
        st.session_state.recommended_questions = []

    #---------------------Displaying Messages---------------------------------------#

    for message in st.session_state.chat_history :
        with st.chat_message(message['role']) :
            st.write(message['content'])

    
    #---------------------------Reading and Storing the PDF in Main--------------------------------#

    if(st.session_state.file is not None and st.session_state.knowledge_base is None) :

        with st.spinner("Processing the Document") :

        #---------------------------Loading Initials--------------------------------#

            with st.spinner("Loading Models") :
                embedding_model = load_embedding_model()

            st.session_state.knowledge_base, st.session_state.chunks = process_pdf(st.session_state.file, embedding_model)

        st.success("Pdf is processed successfully")

    
    #------------------------------Generating Recommended Questions in Main--------------------------#

    
    if(st.session_state.recommended_questions == [] and st.session_state.file is not None) :
        
        with st.spinner("Generating recommended questions...") :
            st.session_state.recommended_questions = generate_recommended_questions(st.session_state.chunks[:5])
    
    elif(st.session_state.recommended_questions) :
        col1, col2, col3 = st.columns(3)

        for i, q in enumerate(st.session_state.recommended_questions) :
            with [col1, col2, col3][i%3] :
                if(st.button(q)) :
                    st.session_state.user_query = q
                    st.rerun()

    #------------------------------Handling the Chat Input----------------------------------#

    prompt = st.chat_input("Ask some questions regarding the document", accept_file=True, file_type = ['pdf'])

    if(prompt) :

        if(prompt.text) :
            st.session_state.user_query = prompt.text

        if(prompt.files) :
            uploaded_file = prompt.files[0]

            if(uploaded_file != st.session_state.file) :
                st.session_state.file = uploaded_file
                st.session_state.knowledge_base = None
                st.session_state.chunks = None
                st.session_state.recommended_questions = []

                st.rerun()

    if(st.session_state.user_query) :
        if(st.session_state.file is None) :
            st.error('Please upload a PDF File')
            st.session_state.user_query = None
        else :
            ask_query()


main()
