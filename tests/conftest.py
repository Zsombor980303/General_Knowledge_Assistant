import pytest

from src.domain.interfaces import EmbeddingProvider, LLMProvider, VectorStoreProvider
from src.domain.models import DocumentChunk, RetrievedChunk


class FakeEmbedder(EmbeddingProvider):
    # Teszteléshez használt mű-embedder: hálózati hívás nélkül fix számsorokat ad vissza.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for _ in texts:
            vectors.append([0.1, 0.2, 0.3])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore(VectorStoreProvider):
    # Memóriában működő mű-vektoradatbázis: eltárolja a mentett chunkokat,
    # és kereséskor az előre beállított találati listát (search_results) adja vissza.
    def __init__(self, search_results: list[RetrievedChunk] | None = None) -> None:
        self.stored_chunks: list[DocumentChunk] = []
        self.search_results: list[RetrievedChunk] = search_results or []

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        self.stored_chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[RetrievedChunk]:
        return self.search_results[:top_k]

    def count(self) -> int:
        return len(self.stored_chunks)


class FakeLLM(LLMProvider):
    # Teszteléshez használt mű-LLM: fix szöveges választ ad vissza, vagy ha beállítunk neki
    # egy kivételt (exception_to_raise), akkor azt dobja el (hibakezelés teszteléséhez).
    def __init__(
        self,
        response_text: str = "Teszt válasz a dokumentumok alapján.",
        exception_to_raise: Exception | None = None,
    ) -> None:
        self.response_text = response_text
        self.exception_to_raise = exception_to_raise
        self.was_called = False
        self.last_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.was_called = True
        self.last_prompt = user_prompt
        if self.exception_to_raise is not None:
            raise self.exception_to_raise
        return self.response_text


@pytest.fixture
def sample_retrieved_chunks() -> list[RetrievedChunk]:
    # Előre elkészített minta találati lista, amelyet bármelyik tesztfüggvény paraméterként átvehet.
    return [
        RetrievedChunk(
            chunk_id="hr_policy.md_0",
            content="Heti 3 nap home office engedélyezett, kivéve a próbaidő alatt.",
            source="hr_policy.md",
            score=0.88,
        ),
        RetrievedChunk(
            chunk_id="it_security.md_1",
            content="A vállalati VPN használata kötelező távmunka esetén.",
            source="it_security.md",
            score=0.75,
        ),
    ]