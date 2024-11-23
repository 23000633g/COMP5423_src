from langchain_core.prompts import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from utils import Utils
from pandas import DataFrame
import pandas as pd

class DataFrameAgent:
    def __init__(self, df: DataFrame):
        self.llm = Utils.getChatModel('gpt-35-turbo')
        self.df = df
        self.pandas_dataframe_agent = create_pandas_dataframe_agent(
            self.llm, self.df, agent_type="openai-tools", verbose=True, allow_dangerous_code=True,
            suffix="Do not create any sample df. Directly use the df provided in the agent. Be concise and helpful."
        )

    def ask(self, query:str) -> dict[str, str]:
        return self.pandas_dataframe_agent.invoke({"input": query})

if __name__ == "__main__":
    df = pd.read_csv("titanic.csv")
    agent = DataFrameAgent(df)
    query = "Who is the oldest passager and what are his/her name and age ?"
    answer = agent.ask(query)
    print(answer)