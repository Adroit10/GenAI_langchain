from dotenv import load_dotenv
import os
import certifi
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ['HF_HOME'] = 'D:/huggingface_cache'
model = ChatHuggingFace.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 128,
        "temperature": 0.7,
    },
)

result = model.invoke("What is the capital of Indonesia")
print(result.content)