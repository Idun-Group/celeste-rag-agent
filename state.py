from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class GraphState(InputState):
    rewritten_query: str
    documents: list[Document]
    answer: str


class OutputState(TypedDict):
    answer: str
