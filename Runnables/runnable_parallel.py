import os
import certifi
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel

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
    template="Generate a tweet on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chains = RunnableParallel(
    {
        'tweet':RunnableSequence(prompt,model,parser),
        'linkedin':RunnableSequence(prompt2,model,parser)
    }
)
result = parallel_chains.invoke({'topic':'AI'})
print(result['tweet'])
print(result['linkedin'])