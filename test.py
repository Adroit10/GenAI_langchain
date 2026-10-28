import langchain

print(langchain.__version__)

from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

p1:Person = {'name':'Dev','age':26}
print(p1.get('name',''))

from pydantic import BaseModel,EmailStr,Field
from typing import Optional
class Student(BaseModel):
    name:str = "Devansh" # default
    age:Optional[int] = None
    email:EmailStr
    cgpa: float = Field(gt=0,lt=10, default=7, description="A decimal value representing the cgpa")


new_student = {'age':32,'email':'tripathidevansh7@gmail.com'} 
student = Student(**new_student)

print(student)