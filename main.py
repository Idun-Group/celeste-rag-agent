from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from idun_agent_engine.prompts import get_prompt

from celeste_adapter import CelesteEmbeddings, call_celeste
from state import GraphState, InputState, OutputState


REWRITE_PROMPT = get_prompt("rewrite_prompt")
RAG_PROMPT = get_prompt("basic_rag")

store: InMemoryVectorStore | None = None


def load_docs(path: str = "docs") -> list[Document]:
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load_and_split()
    print(f"fetched {len(docs)} documents!")
    return docs


async def rewrite_query(state: GraphState) -> dict:
    question = state["messages"][-1].content
    rewritten = await call_celeste(REWRITE_PROMPT.format(question=question))
    return {"rewritten_query": rewritten.strip()}


async def retrieve(state: GraphState) -> dict:
    global store
    if store is None:
        docs = load_docs()
        embeddings = CelesteEmbeddings()
        store = await InMemoryVectorStore.afrom_documents(docs, embeddings)
    query = state["rewritten_query"]
    results = await store.asimilarity_search(query, k=4)
    return {"documents": results}


async def generate(state: GraphState) -> dict:
    question = state["messages"][-1].content
    docs = state["documents"]
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    answer = await call_celeste(RAG_PROMPT.format(context=context, question=question))
    return {"answer": answer.strip()}


def build_graph():
    graph = StateGraph(GraphState, input=InputState, output=OutputState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph


graph = build_graph()
