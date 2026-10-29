import os
import certifi
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


os.environ['SSL_CERT_FILE'] = certifi.where()
model = ChatHuggingFace.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 128,
            "temperature": 0.7,
            "return_full_text": False
        },
)
parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional character from a book. Output ONLY valid JSON, with no markdown formatting, no backticks, and no introductory text \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

#prompt = template.format()
#result = model.invoke(prompt)
#final_result = parser.parse(result.content)

chain = template | model | parser
final_result = chain.invoke({})

print(final_result)
print(type(final_result))