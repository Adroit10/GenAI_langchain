import os 
import certifi
from dotenv import load_dotenv
load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

model = ChatHuggingFace.from_model_id(
    model_id='HuggingFaceTB/SmolLM2-360M-Instruct',
    task='text-generation',
    pipeline_kwargs=
        {
            'temperature':0.1,
            'return_full_text':False
        }
)

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description="Give the sentiment of the feedback")

parser = StrOutputParser()

#parser2 = PydanticOutputParser(pydantic_object=Feedback)
parser2 = StrOutputParser()
#prompt1 = PromptTemplate(
#    template = """Classify the sentiment of the following feedback text as ONLY 'positive' or 'negative'. Do not include any other text\n {feedback}
#    Fill in the schema with actual data values, NOT the schema definition itself.
#    \n {format_instruction}""",
#    input_variables=['feedback'],
#    partial_variables={'format_instruction':parser2.get_format_instructions()}
#)
prompt = PromptTemplate(
    template="Classify the sentiment of the following feedback text as ONLY 'positive' or 'negative'. Do not include any other text.\nFeedback: {feedback}\nSentiment:",
    input_variables=['feedback']
)
classifier_chain = prompt | model | parser2

prompt2 = PromptTemplate(
    template="You are a customer service executive. Write an appropriate short polite response to this positive feedback\n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="You are a customer service executive. Write an appropriate,short empathetic response to this negative feedback\n {feedback}",
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x: 'positive' in x.lower(), prompt2 | model | parser),
    (lambda x: 'negative' in x.lower(), prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | branch_chain
result = chain.invoke({'feedback':'This is a terrible phone'})

print(result)

chain.get_graph().print_ascii()