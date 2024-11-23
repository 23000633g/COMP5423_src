import gradio as gr
from ChatBotAgent import ChatBotAgent 

def chat_with_agent(user_input, history):
    response = agent.ask(user_input)
    return response

agent = ChatBotAgent()

iface = gr.ChatInterface(
    fn=chat_with_agent,  
    title="GloVe Dataset Chatbot",
    description="I am a helpful assistant trained on the Titanic dataset. Ask me any questions about it!"
)

iface.launch()