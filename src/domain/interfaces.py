from abc import ABC, abstractmethod
from src.domain.models import DocumentChunk, RetrievedChunk

# Külső szolgáltatások (embedding, vektoros tároló, LLM) absztrakt interfészei.
#
# Miért így épül fel?
# Ez a fájl csak a "szabványt" (a kötelező metódusneveket és típusokat) írja le,
# maga a függvénytörzs szándékosan üres marad (pass), mint C-ben egy .h header fájlban.
# A tényleges megvalósítás külön fájlokban (az infrastructure/ rétegben) kap helyet,
# például külön osztályként a Google Geminihez és a lokális Ollamához.
#
# Mire jó ez a gyakorlatban?
# 1. Cserélhetőség: A fő program csak ezeket az interfészeket ismeri, így a .env
#    átírásával, kódmódosítás nélkül tudunk váltani felhős és lokális futás között.
# 2. Tesztelhetőség: A tesztekben fizetős API nélkül is könnyen beadhatunk kamu (mock)
#    osztályokat, amelyek ugyanezeket a metódusokat valósítják meg.


class EmbeddingProvider(ABC):
    # Szöveg -> számsor (vektor) átalakítás interfésze.
    # Az ABC + @abstractmethod kikényszeríti, hogy minden leszármazott osztály
    # (GeminiEmbedder, OllamaEmbedder) kötelezően megírja ezt a két metódust.

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Több szövegrészlet vektorizálása egyszerre (adatbázisba töltéshez).
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        # Egyetlen felhasználói kérdés vektorizálása (kereséshez).
        pass


class VectorStoreProvider(ABC):
    # Vektoros adatbázis interfésze (nálunk a ChromaDB valósítja meg).

    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        # Dokumentumrészletek és a hozzájuk tartozó vektorok mentése.
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        # A kérdéshez legjobban hasonlító top_k darab részlet visszakeresése.
        pass

    @abstractmethod
    def count(self) -> int:
        # Tárolt dokumentumrészletek száma (pl. hogy induláskor ne töltsük be őket duplán).
        pass


class LLMProvider(ABC):
    # Szöveggeneráló AI modell interfésze (Gemini és Ollama is ezt valósítja meg).

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Rendszerutasítás + kontextussal dúsított kérdés alapján visszaadja az AI válaszát.
        pass