from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama  # swap for OpenAI if preferred
from modules.retriever import retrieve
from modules.embeddings import VectorStore
llm = Ollama(model='phi3:mini')
vs = VectorStore()

chunks = [
    'The RAG pipeline retrieves context before generating answers.',
    'FAISS provides sub-linear nearest-neighbour search.'
]

vs.add_chunks(chunks)
RAG_PROMPT = PromptTemplate(
    input_variables=['context', 'question'],
    template=(
        'You are a helpful assistant. Answer ONLY using the context below.\n'
        'If the context does not contain the answer, say "I don\'t know."\n\n'
        'Context:\n{context}\n\nQuestion: {question}\n\nAnswer:'
    )
)

def rag_answer(question):
    hits = retrieve(question, vs, k=4)
    context = '\n---\n'.join(h['chunk'] for h in hits)
    prompt = RAG_PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt)
    sources = [h['chunk'][:80] + '...' for h in hits]
    return {'answer': answer.strip(), 'sources': sources}

if __name__ == '__main__':
    result = rag_answer('What is retrieval-augmented generation?')
    print('Answer:', result['answer'])
    print('Sources:', result['sources'])
