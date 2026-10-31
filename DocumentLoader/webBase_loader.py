from langchain_community.document_loaders import WebBaseLoader
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
            "temperature": 0.7,
            'return_full_text':False
        },
)

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=['question','text']
)


url = r'https://www.flipkart.com/apple-2020-macbook-air-m1-16-gb-256-gb-ssd-mac-os-big-sur-mgn63hn-a/p/itmec35accf62a2f?pid=COMH2SZFDTJZFD94&lid=LSTCOMH2SZFDTJZFD94Z0M86Y&marketplace=FLIPKART&cmpid=content_computer_8965229628_gmc'
loader = WebBaseLoader(url)

docs=loader.load()

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({'question':'What is the memory specification of the product','text':docs[0].page_content}))
