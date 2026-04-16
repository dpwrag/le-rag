import getpass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


model = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_template("""
Role: You are a helpful assistant that translates English to French.
User: {text}
""")

chain = prompt | model

ai_msg = chain.invoke({"text": "I love programming."})
print(ai_msg.content)
