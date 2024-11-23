from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

class Utils:
    @staticmethod
    def getChatModel(model='gpt-4o-mini'):
        if (model == 'gpt-35-turbo'):
            return ChatOpenAI(model=model)
        return ChatOpenAI(model=model)


        
    @staticmethod
    def getEmbeddingModel(model='text-embedding-3-large'):
        return OpenAIEmbeddings(model=model)
