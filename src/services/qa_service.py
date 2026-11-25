from typing import Any

from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.domain.models import RetrievedChunk

logger = get_logger(__name__)

# Miért van szükség erre a fájlra a webes végpont (routes.py) és a LangGraph (workflow.py) között?
# 1. Bemenet-ellenőrzés (Validáció): Ha valaki üres kérdést vagy csak szóközöket küld be, itt azonnal
#    megfogjuk (ValidationError), így nem égetünk el feleslegesen API-hívást az AI felé.
# 2. Elrejti a LangGraph belső működését: A webes rétegnek nem kell ismernie a RAGState szótárt
#    vagy az .invoke() hívást – csak meghívja a qa_service.ask(kérdés) függvényt.
# 3. Letisztított kimenetet ad vissza: A gráf teljes belső állapotából (final_state) kiszedi
#    csak a kész választ ("answer") és a forráshivatkozásokat ("sources") az API (és a tesztek) számára.

class QAService:
    # A kérdés-válasz üzleti logikát fogja össze: ellenőrzi a bemeneti kérdést,
    # elindítja a LangGraph RAG folyamatot, és formázva visszaadja a választ + a forrásokat.

    def __init__(self, rag_graph: Any) -> None:
        # Itt kapja meg a main.py-ban lefordított (compiled) LangGraph objektumot.
        self._rag_graph = rag_graph

    def ask(self, question: str) -> dict[str, Any]:
        # 1. Ellenőrizzük, hogy a kérdés nem üres-e vagy csak szóközökből áll-e.
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValidationError("A kérdés nem lehet üres.")

        logger.info(f"Új kérdés feldolgozása: '{cleaned_question}'")

        # 2. Elindítjuk a LangGraph munkafolyamatot a kezdő állapottal (csak a question van benne).
        # A visszatérő final_state már tartalmazza a megtalált chunkokat és a kész választ is.
        final_state = self._rag_graph.invoke({"question": cleaned_question})

        answer: str = final_state.get("answer", "")
        sources: list[RetrievedChunk] = final_state.get("retrieved_chunks", [])

        # 3. Egységes szótárban adjuk vissza a választ és a forrásrészleteket az API rétegnek.
        return {
            "answer": answer,
            "sources": sources,
        }