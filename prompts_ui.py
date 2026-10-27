from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
import os
import certifi
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate,load_prompt

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

st.header("Research Tool")

paper_input = st.selectbox("Select a Research Paper Name: ",["Select...","Attention is All You Need",
                                                             "BERT:Pre-Training of Deep Bidirectional Transformers","GPT-3: Language Models are Few-Shot Learners",
                                                             "Diffusion Models Beat GANs on Image Synthesis"
                                                             ])
style_input = st.selectbox("Select Explanation Style",["Beginner-Friendly","Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox("Select Explanation Length",["Short (1-2 paragraphs)", "Medium (3-5 Paragraphs)", "Long (detailed explanation)"])

template = load_prompt('template.json')




if st.button('Summarize'):

    chain = template | model
    result =chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
}
    )
    st.write(result.content)
else:
    st.write("Please enter a prompt")