from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router
from src.core.config import Settings
from src.core.exceptions import LLMTimeoutError
from src.graphs.workflow import build_rag_graph
from src.services.qa_service import QAService
from tests.conftest import FakeEmbedder, FakeLLM, FakeVectorStore


def _build_test_client(llm: FakeLLM, search_results: list) -> TestClient:
    """Összeállít egy könnyűsúlyú FastAPI tesztalkalmazást valódi külső hálózati hívások nélkül."""
    app = FastAPI()
    settings = Settings(gemini_api_key="test-key")
    graph = build_rag_graph(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(search_results=search_results),
        llm=llm,
        top_k=3,
        min_score=0.25,
    )
    app.state.settings = settings
    app.state.qa_service = QAService(rag_graph=graph)
    app.include_router(router)
    return TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    """Ellenőrzi a GET /health állapotellenőrző végpont sikeres válaszát."""
    client = _build_test_client(llm=FakeLLM(), search_results=[])
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_ask_endpoint_returns_answer_and_sources(sample_retrieved_chunks) -> None:
    """Ellenőrzi a sikeres POST /ask végpontot: visszaadja a generált választ és a forráshivatkozásokat."""
    client = _build_test_client(
        llm=FakeLLM(response_text="A VPN használata kötelező."),
        search_results=sample_retrieved_chunks,
    )
    response = client.post("/ask", json={"question": "Kötelező a VPN?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A VPN használata kötelező."
    assert len(body["sources"]) == 2
    assert body["sources"][0]["source"] == "hr_policy.md"


def test_ask_endpoint_rejects_whitespace_only_question_with_400() -> None:
    """Ellenőrzi, hogy a csak szóközökből álló kérdést a szolgáltatás HTTP 400-as hibával elutasítja."""
    client = _build_test_client(llm=FakeLLM(), search_results=[])
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 400


def test_ask_endpoint_rejects_empty_string_with_422() -> None:
    """Ellenőrzi, hogy a teljesen üres stringet már a Pydantic séma elutasítja HTTP 422-vel."""
    client = _build_test_client(llm=FakeLLM(), search_results=[])
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_endpoint_returns_504_on_llm_timeout(sample_retrieved_chunks) -> None:
    """Ellenőrzi, hogy ha az LLM időtúllépést dob, a végpont HTTP 504 Gateway Timeout hibát ad vissza."""
    failing_llm = FakeLLM(exception_to_raise=LLMTimeoutError("Időtúllépés"))
    client = _build_test_client(llm=failing_llm, search_results=sample_retrieved_chunks)

    response = client.post("/ask", json={"question": "Hány nap szabadság jár?"})

    assert response.status_code == 504