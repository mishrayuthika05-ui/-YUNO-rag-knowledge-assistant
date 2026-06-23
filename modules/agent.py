# ===== Module 5: Tool-Using Agent =====
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
import numexpr

llm = Ollama(model='phi3:mini')

def safe_eval(expr):
    """Evaluate a simple math expression safely with numexpr."""
    try:
        return str(numexpr.evaluate(expr.strip()).item())
    except Exception as e:
        return f'Error: {e}'

calculator_tool = Tool(
    name='Calculator',
    func=safe_eval,
    description='Evaluate a mathematical expression. Input must be a valid math string.'
)

search_tool = DuckDuckGoSearchRun()
web_tool = Tool(
    name='WebSearch',
    func=search_tool.run,
    description='Search the web for current information. Input is a search query.'
)

agent = initialize_agent(
    tools=[calculator_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == '__main__':
    q = 'What is the square root of 144?'
    print(agent.run(q))
