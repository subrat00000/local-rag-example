from langchain_community.vectorstores.qdrant import Qdrant
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams,Distance
from ollama import Client
from fastapi import FastAPI, File, UploadFile, HTTPException,BackgroundTasks
import os

app = FastAPI()

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    ollama_url = "https://7d7b-34-121-115-168.ngrok-free.app"
    qdrant_url= "https://2f855f0f-1a16-435e-b0c6-6778fab46d1b.us-east4-0.gcp.cloud.qdrant.io"
    qdrant_api_key="WiPEu3ZkXJg-n2KH9nhEZtPvvrWq9VOBUmQy9VS2Tb8TZzY5bFAQ2w"

    def __init__(self):
        ollama = Client(host=self.ollama_url)
        if "mistral:latest" in [models["name"] for models in ollama.list()["models"]] :
            pass
        else:
            ollama.pull("mistral")
        self.model = ChatOllama(model="mistral",base_url=self.ollama_url)
        self.client = QdrantClient(url=self.qdrant_url,api_key=self.qdrant_api_key)
        self.qdrant = Qdrant(client=self.client,collection_name="mosaic",embeddings=self.embeddings)
        collections=[]
        for data in self.client.get_collections().collections:
            collections.append(data.name)
        if "mosaic" in collections:
            pass
        else:
            self.client.create_collection(collection_name="mosaic",vectors_config=VectorParams(size=768, distance=Distance.COSINE),)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are a helper for question answering tasks. Use the following context to answer the question.
            If you don't know the answer, just say you don't know. Use three sentences
            maximum and be concise in your response. [/INST] </s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        Qdrant.add_documents(self=self.qdrant,documents=chunks)

        

    def ask(self, query: str):
        self.retriever = self.qdrant.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
            },
        )
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

chat_pdf = ChatPDF()

@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        os.makedirs("temp", exist_ok=True)
        file_path = f"temp/{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        background_tasks.add_task(chat_pdf.ingest(file_path))
        
        return {"filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/ask")
async def ask_question(query: str):
    response = chat_pdf.ask(query)
    return {"response": response}

@app.get("/clear/")
async def clear_cache():
    chat_pdf.clear()
    return {"message": "Cache cleared successfully."}