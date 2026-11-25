import pytest
from fastapi.testclient import TestClient

from src.main import create_app


@pytest.mark.live
def test_live_provider_end_to_end_ask() -> None:
    """Valódi konfigurált AI szolgáltatóval (Gemini vagy lokális Ollama) futó integrációs próba.

    Futtatás külön: pytest -m live -v
    """
    app = create_app()
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200

    ask_resp = client.post("/ask", json={"question": "Hány nap home office engedélyezett hetente?"})
    assert ask_resp.status_code == 200
    body = ask_resp.json()
    assert len(body["answer"]) > 0
    assert len(body["sources"]) > 0