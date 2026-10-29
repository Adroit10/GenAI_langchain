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
            "return_full_text": False
        },
)

prompt1 = PromptTemplate(
    template = "Generate a detailed report on {topic}. Just give me the output related to the topic and no other information",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "generate a 5 pointer summary on the following text \n {text}",
    input_variables=['text']
)
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Athena'})
print(result)

chain.get_graph().print_ascii()
