import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ... other imports ...

# ADD THESE TWO LINES (replace with your actual key):
os.environ["OPENAI_API_KEY"] = ""

# ... rest of your code ...

# 1. Load Documents
# We use WebBaseLoader to scrape a blog post. bs4_strainer ensures we only get the relevant text.
print("Loading documents...")
loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Bruno_Fernandes"),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# 2. Split Documents
# LLMs have context windows, so we chunk the text into smaller, manageable pieces.
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    add_start_index=True
)
splits = text_splitter.split_documents(docs)

# 3. Index & Embed (Vector Store)
# We convert our text chunks into numerical vectors (embeddings) and store them in ChromaDB.
print("Embedding and indexing chunks...")
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)

# Create a retriever that returns the top 6 most similar chunks for a given query
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# 4. Define the RAG Prompt
# You can pull a prompt from LangChain Hub (hub.pull("rlm/rag-prompt")) or define your own.
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

# 5. Build the Generation Chain using LCEL (LangChain Expression Language)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Helper function to format the retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("Building the RAG chain...")
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# 6. Execute the Pipeline
if __name__ == "__main__":
    question = "What is Task Decomposition?"
    print(f"\nQuestion: {question}")
    
    # Invoke the chain
    answer = rag_chain.invoke(question)
    
    print("\nAnswer:")
    print(answer)