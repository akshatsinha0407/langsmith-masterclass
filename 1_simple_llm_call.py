from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Same one-line prompt
prompt = PromptTemplate.from_template("{question}")

# Gemini 2.5 Flash model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

parser = StrOutputParser()

# Same chain: prompt → model → parser
chain = prompt | model | parser

# Run
result = chain.invoke({"question": "What is the capital of India?"})
print(result)
