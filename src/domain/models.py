from dataclasses import dataclass, field

# Ez a fájl a rendszer saját belső adatstruktúráit (itt konkrétan struct) tartalmazza.
# Azért itt definiáljuk őket, mert egyik külső könyvtártól (ChromaDB, LangChain, FastAPI) sem függnek,
# így a program összes rétege ezen a közös adatformátumon keresztül kommunikál.
# A @dataclass automatikusan legyártja a konstruktort (__init__) és az alap metódusokat
# A frozen=True tulajdonképpen constanst jelent


@dataclass(frozen=True)
class DocumentChunk:
    # Egy feldarabolt dokumentumrészlet, amit betöltünk a vektoros adatbázisba.
    # A frozen=True azt jelenti, hogy létrehozás után már nem lehet véletlenül módosítani (read-only).
    chunk_id: str # Egyedi azonosító
    content: str # A kivágott szövegrészlet a dokumentumból
    source: str # A forrásfájl neve
    metadata: dict[str, str | int] = field(default_factory=dict) # szótár-kulcs érték, bármilyen extra információnak


@dataclass(frozen=True)
class RetrievedChunk:
    # Egy keresési találat, amit a vektoros adatbázis ad vissza a kérdés alapján.
    # Tartalmazza a szöveget, a forrásfájl nevét (hogy hivatkozni tudjunk rá) és a hasonlósági pontszámot.
    chunk_id: str # A megtalált szövegrészlet azonosítója
    content: str # A megtalált szövegrészlet szövege
    source: str # A forrásfájl neve
    score: float # Relevancia/hasonlósági pontszám (mennyire passzol a kérdéshez)