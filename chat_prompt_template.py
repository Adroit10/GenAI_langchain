from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage,SystemMessage
chat_template = ChatPromptTemplate.from_messages([
    ("system","You are a helpful {domain} expert"),
    ("human","Explain in simple terms, what is {topic}")
]
)

prompt = chat_template.invoke({'domain':'football','topic':'offside'})
print(prompt)
