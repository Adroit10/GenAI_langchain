from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
import os
import certifi
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
model = ChatHuggingFace.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 128,
        "temperature": 0.7,
        "return_full_text": False
    },
)

chat_history = [
    SystemMessage(content="You are a Helpful AI agent"),

]


while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content =  result.content))
    print("AI: ",result.content)

print(chat_history)