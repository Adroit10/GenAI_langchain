from langchain_community.document_loaders import TextLoader
import os
import certifi
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ['SSL_CERT_FILE'] = certifi.where()
model = ChatHuggingFace.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 128,
            "temperature": 0.7,
        },
)


loader = TextLoader(r'DocumentLoader\StoppingByTheWoods.txt')

docs = loader.load()

prompt = PromptTemplate(
    template="Write a summary for the followwing poem - \n{poem}",
    input_variables=['poem']
)

parser =StrOutputParser()

chain = prompt | model | parser
print(docs[0].page_content)
print(docs[0].metadata)
print(chain.invoke({'poem':docs[0].page_content}))