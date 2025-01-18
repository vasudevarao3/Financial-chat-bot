import time
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
import google.generativeai as genai
import os


class APIKeysManager:
    """Manages API keys for the application."""

    def __init__(self):
        load_dotenv()
        self.keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
            "LLAMAPARSER_API_KEY": os.getenv("LLAMAPARSER_API_KEY"),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        }
        self.validate_keys()

    def validate_keys(self):
        missing_keys = [key for key, value in self.keys.items() if not value]
        if missing_keys:
            raise ValueError(f"Missing API keys: {', '.join(missing_keys)}")

    def get_key(self, key_name):
        return self.keys.get(key_name)
    

class DocumentManager:
    """Manages document loading, splitting, and embedding."""

    def __init__(self, llama_parser_api_key, embedding_model):
        self.loader = LlamaParse(api_key=llama_parser_api_key, result_type="markdown", verbose=True)
        self.splitter = SentenceSplitter(include_metadata=True)
        self.embedding_model = embedding_model

    def process_documents(self, input_files):
        print("Loading and processing documents...")
        documents = self.loader.load_data(file_path=input_files)
        nodes = self.splitter.get_nodes_from_documents(documents)
        for node in nodes:
            if node.text:
                node.embedding = self.embedding_model.get_text_embedding(node.get_content(metadata_mode="all"))
        return nodes


class PineconeManager:
    """Manages Pinecone interactions."""

    def __init__(self, api_key, index_name, dimension=1536, metric="cosine", region="us-east-1"):
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.region = region
        self.client = PineconeGRPC(api_key=api_key)
        self.index = self._initialize_index()

    def _initialize_index(self):
        print("Initializing Pinecone index...")
        if self.index_name not in self.client.list_indexes().names():
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud="aws", region=self.region),
            )
            while not self.client.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
        return self.client.Index(self.index_name)

    def get_vector_store(self):
        return PineconeVectorStore(pinecone_index=self.index)
    

class QueryEngineManager:
    """Manages query engine setup and operations."""

    def __init__(self, pinecone_manager, embedding_model):
        self.vector_store = pinecone_manager.get_vector_store()
        self.vector_index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        self.retriever = VectorIndexRetriever(index=self.vector_index, similarity_top_k=7)
        self.postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever, node_postprocessors=[self.postprocessor]
        )

    def query(self, query):
        print("Running query on the query engine...")
        return self.query_engine.query(query)
    

class QueryRewriter:
    """Handles query rewriting based on history."""

    def __init__(self, openai_api_key):
        self.llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)

    def rewrite_query(self, query, history=""):
        query_prompt = f"""
            ### Instruction: Rewrite the query

            The original query is as follows: {query}
            We have provided an existing history: {history}
            
            Rewrite the query into a clear and specific form suitable for a vector database search.
            - Preserve the original meaning.
            - Use history only if necessary.
        """
        rewritten_query = self.llm.complete(query_prompt)
        return rewritten_query.text
    

class MultiIndexManager:
    """Manages multiple Pinecone indices for different contexts."""

    def __init__(self, api_key, indices_info):
        self.api_key = api_key
        self.indices = {}
        self.client = PineconeGRPC(api_key=api_key)

        for index_name, config in indices_info.items():
            if index_name not in self.client.list_indexes().names():
                self.client.create_index(
                    name=index_name,
                    dimension=config['dimension'],
                    metric=config['metric'],
                    spec=ServerlessSpec(cloud="aws", region=config['region']),
                )
            self.indices[index_name] = self.client.Index(index_name)

    def get_vector_store(self, index_name):
        return PineconeVectorStore(pinecone_index=self.indices[index_name])
    

class ContextAugmentedQueryEngine:
    """Manages query processing across multiple indices."""

    def __init__(self, multi_index_manager, embedding_model):
        self.multi_index_manager = multi_index_manager
        self.embedding_model = embedding_model

    def query_index(self, index_name, query, similarity_cutoff=0.3):
        vector_store = self.multi_index_manager.get_vector_store(index_name)
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)
        postprocessor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        query_engine = RetrieverQueryEngine(
            retriever=retriever, node_postprocessors=[postprocessor]
        )
        return query_engine.query(query)

    def query_all_indices(self, query, similarity_cutoff=0.3):
        results = {} 
        for index_name in self.multi_index_manager.indices.keys():
            print(f"Querying index: {index_name}")
            results[index_name] = self.query_index(index_name, query, similarity_cutoff)
        return results


# Simle RAG when we have single vector store
class RAGApplication:
    """Main application class to manage the entire workflow."""

    def __init__(self):
        print("Initializing API keyss")
        self.api_keys = APIKeysManager()
        print("API Keys verified")
        self.embedding_model = OpenAIEmbedding(model="text-embedding-3-small")
        print("Embedding model initialized")
        Settings.embed_model = self.embedding_model
        print("Embedding model set")
        self.document_manager = DocumentManager(
            llama_parser_api_key=self.api_keys.get_key("LLAMAPARSER_API_KEY"),
            embedding_model=self.embedding_model,
        )
        print("Document manager initialized")
        self.pinecone_manager = PineconeManager(
            api_key=self.api_keys.get_key("PINECONE_API_KEY"),
            index_name="capston-llama-parser"
        )
        self.pinecone_manager = PineconeManager(
            api_key=self.api_keys.get_key("PINECONE_API_KEY"),
            index_name="alteryx-llama-parser"
        )
        self.pinecone_manager = PineconeManager(
            api_key=self.api_keys.get_key("PINECONE_API_KEY"),
            index_name="infosys-llama-parser"
        )
        self.pinecone_manager = PineconeManager(
            api_key=self.api_keys.get_key("PINECONE_API_KEY"),
            index_name="uber-llama-parser"
        )
        print("Pinecone manager initialized")
        self.query_engine_manager = QueryEngineManager(
            pinecone_manager=self.pinecone_manager, embedding_model=self.embedding_model
        )
        print("Query engine manager initialized")
        self.query_rewriter = QueryRewriter(openai_api_key=self.api_keys.get_key("OPENAI_API_KEY"))
        print("Query rewriter initialized")

    def run_query(self, query, history=""):
        relavent_documents = []
        rewritten_query = self.query_rewriter.rewrite_query(query, history)
        response = self.query_engine_manager.query(rewritten_query)
        pprint_response(response, show_source=True)
        for node in response.source_nodes:
            relavent_documents.append(node.text)
        return relavent_documents, rewritten_query

    def upload_documents(self, input_files):
        print("Uploading documents...")
        nodes = self.document_manager.process_documents(input_files)
        vector_store = self.pinecone_manager.get_vector_store()
        vector_store.add(nodes=nodes)
        print("Documents uploaded successfully.")


class RAGApplicationWithAgents(RAGApplication):
    """Extended RAG application with multiple indices and OpenAI function-calling agents."""

    def __init__(self):
        super().__init__()

        indices_info = {
            "uber-llama-parser": {"dimension": 1536, "metric": "cosine", "region": "us-east-1"},
            "infosys-llama-parser": {"dimension": 1536, "metric": "cosine", "region": "us-east-1"},
            "alteryx-llama-parser": {"dimension": 1536, "metric": "cosine", "region": "us-east-1"},
        }

        self.multi_index_manager = MultiIndexManager(
            api_key=self.api_keys.get_key("PINECONE_API_KEY"), indices_info=indices_info
        )
        self.context_query_engine = ContextAugmentedQueryEngine(
            multi_index_manager=self.multi_index_manager, embedding_model=self.embedding_model
        )

    def run_context_aware_query(self, query, history = ""):
        rewritten_query = self.query_rewriter.rewrite_query(query, history)
        results = self.context_query_engine.query_all_indices(rewritten_query)
        aggregated_results = {}
        for index, response in results.items():
            documents = [node.text for node in response.source_nodes]
            aggregated_results[index] = {
                "documents": documents,
                "response": response.response,
                "metadata": response.metadata,
            }
        return results, aggregated_results, rewritten_query

    
if __name__=="__main__":
    app = RAGApplicationWithAgents()
    print("Application initialized")
    query = "What is the revenue of Infosys in 2020?"
    print("Running query...")
    history = ""
    results, aggregated_results, rewritten_query = app.run_context_aware_query(query, history)
    print("Rewritten Query:", rewritten_query)
    print("Results:", results)
    print("Aggregated Results:", aggregated_results)
    
