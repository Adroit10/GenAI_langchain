from langchain_huggingface import HuggingFaceEmbeddings
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is capital of India"
vector = embedding.embed_query(text)
print(vector)