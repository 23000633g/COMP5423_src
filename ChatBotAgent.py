from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver 
import pandas as pd
from DataFrameAgent import DataFrameAgent
from NounVectorStore import NounVectorStore
from utils import Utils

class ChatBotAgent:
    def __init__(self):
        name_retriever = NounVectorStore.createRetrieverTool(pd.read_csv('titanic.csv'), field='Name')
        self.tools = [self.ask_df_agent, name_retriever]
        self.model = Utils.getChatModel()
        self.system_message = """You are a helpful assistant "GloVe Dataset Chatbot". 
        All passager name must ALWAYS first look up the filter value using the "search_proper_nouns" tool before using it in a query.
        If user ask for info on Titanic dataset(titanic.csv) directly pass the question to ask_df_agent. 
        Do not provide any code or guide to get the info. Be concise and helpful. 
        You can provide the info directly or ask for more details if needed. 
        Do not try to guess at the proper name - use this function to find similar ones"""
        self.memory = MemorySaver()
        self.langgraph_agent_executor = create_react_agent(
                self.model,  self.tools, state_modifier= self.system_message, 
                checkpointer= self.memory
        )
        self.config = {"configurable": {"thread_id": "test-thread"}}

    @tool()
    def ask_df_agent(query: str, dataset: str) -> str:
        """Direction question on Titanic dataset(titanic.csv) to this tool"""
        df = pd.read_csv(dataset)
        df_agent = DataFrameAgent(df)
        return df_agent.ask(query)
    
    def ask(self, query: str):
        return self.langgraph_agent_executor.invoke({"messages": [("user", query)]}, self.config)["messages"][-1].content
           
if __name__ == "__main__":
    agent = ChatBotAgent()
    query = "Hi, I'm GloVe! What is the average age of the passagers?"
    print(agent.ask(query))
    print("---")
    query = "What is the name of oldest passenger and how old is he/she?"
    print(agent.ask(query))
    print("---")
    query = "Is there any correlation between gender and survival ? Which gender has the higher survival rate ?"
    print(agent.ask(query))
    print("---")
    query = "Did Mrs. Jacques Heath survived ?"
    print(agent.ask(query))
    print("---")
    query = "Did Mr. Jacqus He**ath survived ?"
    print(agent.ask(query))
    print("---")
    query = "Remember my name?"
    print(agent.ask(query))
    print("---")
    query = "Remember what did i asked?"
    print(agent.ask(query))
    print("---")
    query = "Remember what were the answer ?"
    print(agent.ask(query))
    print("---")