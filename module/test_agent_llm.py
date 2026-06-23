from langchain_community.llms import Ollama

llm = Ollama(model="phi3:mini")

prompt = """
Answer only with the result.

Question: What is the square root of 144?
"""

response = llm.invoke(prompt)

print(response)
