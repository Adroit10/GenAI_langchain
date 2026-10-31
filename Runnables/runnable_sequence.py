import os
import certifi
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

os.environ['SSL_CERT_FILE'] = certifi.where()
model = ChatHuggingFace.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 128,
            "temperature": 0.7,
        },
)
prompt = PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text} ",
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt,model,parser,prompt2,model,parser)

print(chain.invoke({'topic':'AI'}))