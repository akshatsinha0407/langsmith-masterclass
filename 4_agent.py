from dotenv import load_dotenv
load_dotenv()

import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


# ---------------- TOOLS ----------------

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city
    """
    url = f"https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}"
    response = requests.get(url, timeout=10)
    return response.text


# ---------------- LLM (Gemini) ----------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)


# ---------------- PROMPT ----------------

prompt = hub.pull("hwchase17/react")


# ---------------- AGENT ----------------

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5
)


# ---------------- RUN ----------------

response = agent_executor.invoke(
    {"input": "What is the current temperature of Gurgaon?"}
)

print(response["output"])
