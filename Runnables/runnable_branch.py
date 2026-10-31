import os
import certifi
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel,RunnableLambda,RunnableBranch

os.environ['SSL_CERT_FILE'] = certifi.where()
model = ChatHuggingFace.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "temperature": 0.7,
            "return_full_text":False
        },
)
prompt = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt,model,parser)

prompt2 = PromptTemplate(
    template="Summarize the following report - \n{report}",
    input_variables=['report']
)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'topic':'Russia Ukraine war'}))