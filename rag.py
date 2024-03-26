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
from sentence_transformers import SentenceTransformer

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    def __init__(self):
        self.model = ChatOllama(model="mistral",base_url="https://ec7d-34-171-191-121.ngrok-free.app")
        self.client = QdrantClient(url="https://2f855f0f-1a16-435e-b0c6-6778fab46d1b.us-east4-0.gcp.cloud.qdrant.io",api_key="WiPEu3ZkXJg-n2KH9nhEZtPvvrWq9VOBUmQy9VS2Tb8TZzY5bFAQ2w")
        self.qdrant = Qdrant(client=self.client,collection_name="mosaic",embeddings=self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] Vous êtes un assistant pour les tâches de réponse aux questions. Utilisez les éléments de contexte suivants pour répondre à la question. 
            Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.. Utilisez trois phrases
             maximum et soyez concis dans votre réponse. [/INST] </s> 
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
