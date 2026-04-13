import celeste
from langchain_core.embeddings.embeddings import Embeddings


class CelesteEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Use aembed_documents in async context")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("Use aembed_query in async context")

    async def aembed_query(self, text: str) -> list[float]:
        vecs = await celeste.text.embed(
            model="gemini-embedding-001",
            text=text,
            dimensions=768,
        )
        return vecs.content

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [await self.aembed_query(text) for text in texts]


async def call_celeste(prompt: str) -> str:
    response = await celeste.text.generate(  # type: ignore
        prompt,
        provider="google",  # type: ignore
        model="gemini-3-flash-preview",
        max_tokens=2048,
        temperature=0.7,
    )
    return response.content
