import pandas as pd
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from utils import Utils

class NounVectorStore:
    
    @staticmethod
    def query_as_list(df, field):
        if field not in df.columns:
            raise ValueError(f"Field '{field}' not found in DataFrame columns")
        if df.empty:
            raise ValueError("DataFrame is empty")
        return df[field].tolist()

    
    @staticmethod
    def createRetrieverTool(df, field = 'Name'):
        store = LocalFileStore("./cache/")
        underlying_embeddings = Utils.getEmbeddingModel()
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, store, namespace=underlying_embeddings.model
        )
        field = NounVectorStore.query_as_list(df, field)
        vector_db = FAISS.from_texts(field, cached_embedder)
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        return create_retriever_tool(
            retriever,
            name="search_proper_nouns",
            description="""Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is valid proper nouns. Use the noun most similar to the search.""",
        )
        
    
if __name__ == "__main__":
    df = pd.read_csv('titanic.csv')
    retriever_tool = NounVectorStore.createRetrieverTool(df)
    print(retriever_tool.invoke("Lily May Peel"))
    print("=====================================")
    print(retriever_tool.invoke("Banana May Peel"))
    print("=====================================")
    print(retriever_tool.invoke("Jacques"))
    print("=====================================")