from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

documents = [
    "Leo Messi Plays for Argentina",
    "Cristiano Ronaldo plays for Portugal",
    "Neymar Jr plays for Brazil"
]

embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)
result = embedding.embed_documents(documents)
print(str(result))