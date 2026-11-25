from pydantic import BaseModel, Field

# Header file. Ez a fájl definiálja a HTTP API bejövő és kimenő JSON formátumait (Pydantic sémák).
# Olyan, mint a domain/models.py, csak kifejezetten a webes kérések és válaszok automatikus
# ellenőrzésére (validálására) és JSON-né alakítására szolgál.


class AskRequest(BaseModel):
    # A POST /ask végpontra érkező JSON kérés szerkezete.
    # A Field(...) a Pydantic mező-konfigurátora, amivel plusz szabályokat adunk a mezőhöz:
    # - ... (három pont): kötelező mező (nincs alapértelmezett értéke, muszáj megadni a JSON-ban).
    # - min_length=1: legalább 1 karakternek kell lennie (az üres "" stringet azonnal visszadobja).
    # - description: ez a leírás jelenik meg automatikusan a FastAPI /docs tesztfelületén.
    question: str = Field(..., min_length=1, description="A felhasználó kérdése a belső tudásbázishoz.")


class SourceItem(BaseModel):
    # Egyetlen felhasznált dokumentumrészlet (forráshivatkozás) formátuma a JSON válaszban.
    chunk_id: str
    source: str
    content: str
    score: float


class AskResponse(BaseModel):
    # A sikeres POST /ask válasz szerkezete: a generált válasz + a felhasznált források listája.
    answer: str
    sources: list[SourceItem]


class ErrorResponse(BaseModel):
    # Egységes hibaüzenet formátum, ha valami elromlik (pl. validációs hiba, timeout, AI hiba).
    error: str
    detail: str