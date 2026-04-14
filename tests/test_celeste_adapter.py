import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    sys.path.insert(0, str(REPO_ROOT))

    fake_celeste = types.ModuleType("celeste")
    fake_celeste.text = SimpleNamespace(
        embed=AsyncMock(),
        generate=AsyncMock(),
        sync=SimpleNamespace(
            embed=Mock(),
            generate=Mock(),
        ),
    )

    embeddings_module = types.ModuleType("langchain_core.embeddings.embeddings")

    class Embeddings:
        pass

    embeddings_module.Embeddings = Embeddings

    sys.modules["celeste"] = fake_celeste
    sys.modules["langchain_core"] = types.ModuleType("langchain_core")
    sys.modules["langchain_core.embeddings"] = types.ModuleType(
        "langchain_core.embeddings"
    )
    sys.modules["langchain_core.embeddings.embeddings"] = embeddings_module
    sys.modules.pop("celeste_adapter", None)

    return importlib.import_module("celeste_adapter"), fake_celeste


class CelesteAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.module, self.fake_celeste = load_module()
        self.adapter = self.module.CelesteEmbeddings()

    def tearDown(self) -> None:
        for name in [
            "celeste_adapter",
            "celeste",
            "langchain_core",
            "langchain_core.embeddings",
            "langchain_core.embeddings.embeddings",
        ]:
            sys.modules.pop(name, None)
        try:
            sys.path.remove(str(REPO_ROOT))
        except ValueError:
            pass

    async def test_aembed_query_returns_single_vector(self) -> None:
        self.fake_celeste.text.embed.return_value = SimpleNamespace(
            content=[0.1, 0.2, 0.3]
        )

        result = await self.adapter.aembed_query("question")

        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.fake_celeste.text.embed.assert_awaited_once_with(
            text="question",
            model=self.module.EMBEDDING_MODEL,
            dimensions=self.module.EMBEDDING_DIMENSIONS,
        )

    async def test_aembed_documents_uses_single_batch_call(self) -> None:
        texts = ["doc one", "doc two"]
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        self.fake_celeste.text.embed.return_value = SimpleNamespace(content=vectors)

        result = await self.adapter.aembed_documents(texts)

        self.assertEqual(result, vectors)
        self.fake_celeste.text.embed.assert_awaited_once_with(
            text=texts,
            model=self.module.EMBEDDING_MODEL,
            dimensions=self.module.EMBEDDING_DIMENSIONS,
        )

    def test_embed_query_uses_sync_api(self) -> None:
        self.fake_celeste.text.sync.embed.return_value = SimpleNamespace(
            content=[0.5, 0.6]
        )

        result = self.adapter.embed_query("sync question")

        self.assertEqual(result, [0.5, 0.6])
        self.fake_celeste.text.sync.embed.assert_called_once_with(
            text="sync question",
            model=self.module.EMBEDDING_MODEL,
            dimensions=self.module.EMBEDDING_DIMENSIONS,
        )

    def test_embed_documents_uses_sync_batch_api(self) -> None:
        texts = ["doc one", "doc two"]
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        self.fake_celeste.text.sync.embed.return_value = SimpleNamespace(
            content=vectors
        )

        result = self.adapter.embed_documents(texts)

        self.assertEqual(result, vectors)
        self.fake_celeste.text.sync.embed.assert_called_once_with(
            text=texts,
            model=self.module.EMBEDDING_MODEL,
            dimensions=self.module.EMBEDDING_DIMENSIONS,
        )

    async def test_call_celeste_returns_response_content(self) -> None:
        self.fake_celeste.text.generate.return_value = SimpleNamespace(
            content="grounded answer"
        )

        result = await self.module.call_celeste("prompt")

        self.assertEqual(result, "grounded answer")
        self.fake_celeste.text.generate.assert_awaited_once_with(
            "prompt",
            model=self.module.GENERATION_MODEL,
            max_tokens=self.module.GENERATION_MAX_TOKENS,
            temperature=self.module.GENERATION_TEMPERATURE,
        )


if __name__ == "__main__":
    unittest.main()
