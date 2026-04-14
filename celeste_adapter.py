import celeste
from langchain_core.embeddings.embeddings import Embeddings

EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS = 768
GENERATION_MODEL = "gemini-3-flash-preview"
GENERATION_MAX_TOKENS = 2048
GENERATION_TEMPERATURE = 0.7


class CelesteEmbeddings(Embeddings):
    """LangChain embeddings adapter backed by Celeste."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = celeste.text.sync.embed(
            text=texts,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        return response.content

    def embed_query(self, text: str) -> list[float]:
        response = celeste.text.sync.embed(
            text=text,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        return response.content

    async def aembed_query(self, text: str) -> list[float]:
        response = await celeste.text.embed(
            text=text,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        return response.content

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        response = await celeste.text.embed(
            text=texts,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        return response.content


async def call_celeste(prompt: str) -> str:
    response = await celeste.text.generate(
        prompt,
        model=GENERATION_MODEL,
        max_tokens=GENERATION_MAX_TOKENS,
        temperature=GENERATION_TEMPERATURE,
    )
    return response.content
