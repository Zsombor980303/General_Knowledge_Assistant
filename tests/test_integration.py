from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router
from src.core.config import Settings
from src.domain.models import RetrievedChunk
from src.graphs.workflow import build_rag_graph
from src.services.ingestion import ingest_documents
from src.services.qa_service import QAService
from tests.conftest import FakeEmbedder, FakeLLM, FakeVectorStore


def test_full_end_to_end_ingestion_and_ask_flow(tmp_path: Path) -> None:
    """Teljes end-to-end integrációs teszt: dokumentum beolvasása lemezről, darabolás, mentés, LangGraph lekérdezés és POST /ask válasz."""
    sample_doc = tmp_path / "szabalyzat.md"
    sample_doc.write_text(
        "# Szabadságolási szabályzat\n\nMinden munkavállalónak évi 20 nap alapszabadság jár.",
        encoding="utf-8",
    )

    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    # 1. Lépés: Tudásbázis betöltése az ideiglenes mappából.
    loaded_count = ingest_documents(
        data_dir=str(tmp_path),
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=500,
        chunk_overlap=100,
    )
    assert loaded_count == 1
    assert len(vector_store.stored_chunks) == 1

    # 2. Lépés: A mentett darab elérhetővé tétele a kereséshez, és a teljes alkalmazás összerakása.
    stored = vector_store.stored_chunks[0]
    vector_store.search_results = [
        RetrievedChunk(
            chunk_id=stored.chunk_id,
            content=stored.content,
            source=stored.source,
            score=0.91,
        )
    ]

    llm = FakeLLM(response_text="Évi 20 nap alapszabadság jár minden munkavállalónak.")
    graph = build_rag_graph(embedder=embedder, vector_store=vector_store, llm=llm, top_k=3, min_score=0.25)

    app = FastAPI()
    app.state.settings = Settings(gemini_api_key="test-key")
    app.state.qa_service = QAService(rag_graph=graph)
    app.include_router(router)

    # 3. Lépés: Valódi HTTP POST kérés küldése a /ask végpontra.
    client = TestClient(app)
    response = client.post("/ask", json={"question": "Hány nap alapszabadság jár?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Évi 20 nap alapszabadság jár minden munkavállalónak."
    assert data["sources"][0]["source"] == "szabalyzat.md"