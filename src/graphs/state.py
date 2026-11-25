from typing import TypedDict
from src.domain.models import RetrievedChunk

# Header file. A LangGraph munkafolyamat közös "állapotcsomagja" (TypedDict = típusos szótár, kb. mint C-ben egy struct).
#
# Hogyan utazik végig a rendszeren (input -> output)?
# 1. Kezdő input (services/qa_service.py): Létrehozza ezt a csomagot csak a "question" mezővel.
# 2. Köztes lépések (graphs/workflow.py): A "retrieve" lépés beleírja a "retrieved_chunks" listát,
#    majd a "generate" (vagy "fallback") lépés beleírja a kész "answer" szöveget.
# 3. Végső output (services/qa_service.py): Visszakapja a teljesen feltöltött állapotcsomagot,
#    és ebből adja vissza a választ és a forráshivatkozásokat az API-nak.


class RAGState(TypedDict, total=False):
    # A total=False azt jelenti, hogy nem kötelező az összes mezőt egyszerre megadni létrehozáskor,
    # hiszen induláskor még csak a "question" létezik, a többi mező csak később töltődik fel.
    question: str                           # Bemenet: a felhasználó kérdése
    retrieved_chunks: list[RetrievedChunk]  # 1. lépés kimenete: a megtalált dokumentumrészletek
    answer: str                             # 2. lépés kimenete: a végső generált válasz