from langchain_community.llms import Ollama

llm = Ollama(model="phi3:mini")

response = llm.invoke("What is 2 + 2?")

print(response)
