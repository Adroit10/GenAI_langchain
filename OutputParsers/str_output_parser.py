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

# detailed report

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

#summary
template2 = PromptTemplate(
    template="Write a five line summary on the following text. \n {text}",
    input_variables=['text']
)

#prompt1 = template1.invoke({'topic':'black hole'})
#result = model.invoke(prompt1)

#prompt2 = template2.invoke({'text':result.content})

#result1= model.invoke(prompt2)
#print(result1.content)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})
print(result)