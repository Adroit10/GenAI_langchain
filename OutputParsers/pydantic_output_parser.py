import os
import certifi
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

os.environ['SSL_CERT_FILE'] = certifi.where()
model = ChatHuggingFace.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 128,
            "temperature": 0.5,
            "return_full_text": False
        },
)


class Person(BaseModel):
    name:str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city:str = Field(description="Name of the city the person belongs to")


parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template="""Generate the name, age, and city of a fictional {region} person.
Fill in the schema with actual data values, NOT the schema definition itself.

Example output format:
{{
  "name": "Rohan Sharma",
  "age": 28,
  "city": "Delhi"
}}

{format_instructions}
""",
    input_variables=['region'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)
#chain = template | model | parser
#final_result = chain.invoke({'region':'Indian'})
prompt = template.invoke({'region':'Indian'})
print(prompt)
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)