from flask import Flask, render_template, request, jsonify, session
import warnings
warnings.filterwarnings("ignore")
import os
import uuid
import markdown
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# --- CONFIGURATION ---
VECTOR_STORE_PATH = "vectorstore_gemini"
NOTES_PATH = "notes"
LLM_MODEL = "gemini-1.5-flash-latest"

# Global variables to store the RAG chain
rag_chain = None
is_initialized = False

def setup_api_key():
    """Sets up the Google API key."""
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "AIzaSyCVoj6CXPvv_kKP4A4U0iIT4UrItE3XIis"  # Replace with your actual key
    return True

def create_or_load_vector_store():
    """Creates or loads the vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    loader = DirectoryLoader(NOTES_PATH, glob="**/*.txt")
    docs = loader.load()
    
    if not docs:
        dummy_doc = [Document(page_content="This is a placeholder. Add your own notes.")]
        return FAISS.from_documents(dummy_doc, embeddings)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

def create_rag_chain(vector_store):
    """Creates the RAG chain using Google Gemini."""
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, convert_system_message_to_human=True)
    
    prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the user's question as helpfully and accurately as possible.
If the answer isn't directly in the context, feel free to suggest ideas or related insights.
Format your response with proper line breaks and use markdown formatting where appropriate.
Use bullet points or numbered lists when listing items.

Context:
{context}

Question: {input}
""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def process_markdown(text):
    """Convert markdown text to HTML."""
    md = markdown.Markdown(extensions=['nl2br', 'tables', 'fenced_code'])
    return md.convert(text)

def initialize_chatbot():
    """Initialize the chatbot components."""
    global rag_chain, is_initialized
    try:
        setup_api_key()
        vector_store = create_or_load_vector_store()
        rag_chain = create_rag_chain(vector_store)
        is_initialized = True
        return True
    except Exception as e:
        print(f"Initialization error: {e}")
        return False

@app.route('/')
def index():
    """Main chat page."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    global rag_chain, is_initialized
    
    if not is_initialized:
        if not initialize_chatbot():
            return jsonify({'error': 'Failed to initialize chatbot. Please check your API key and notes directory.'}), 500
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        show_sources = data.get('show_sources', False)
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get response from RAG chain
        response = rag_chain.invoke({"input": user_message})
        
        # Process the response text to convert markdown to HTML
        processed_response = process_markdown(response["answer"])
        
        result = {
            'response': processed_response
        }
        
        # Only include sources if requested
        if show_sources:
            result['sources'] = [
                {
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'source': doc.metadata.get('source', 'N/A')
                }
                for doc in response.get("context", [])
            ]
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/status')
def status():
    """Check if the chatbot is ready."""
    return jsonify({'initialized': is_initialized})

if __name__ == '__main__':
    os.makedirs(NOTES_PATH, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
