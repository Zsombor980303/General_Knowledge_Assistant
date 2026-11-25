from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from src.core.exceptions import ConfigurationError, LLMServiceError
from src.core.logging import get_logger
from src.domain.interfaces import EmbeddingProvider

logger = get_logger(__name__)


class GeminiEmbedder(EmbeddingProvider):
    # Az absztrakt EmbeddingProvider interfész konkretizált, felhős (Google Gemini) megvalósítása.
    # Mivel ez már nem absztrakt osztály, ebből fogunk példányt létrehozni, ha Gemini van beállítva.

    def __init__(self, api_key: str, model_name: str) -> None:
        # Példányosításkor itt még csak bekonfiguráljuk a klienst és eltesszük a self._client
        # belső változóba, tényleges hálózati API-hívás itt még nem történik.
        if not api_key:
            raise ConfigurationError("A Gemini embedding használatához kötelező a GEMINI_API_KEY megadása.")
        self._client = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Tényleges API-hívás: a bemeneti szöveglistából (texts) vektorokat készít adatbázisba töltéshez.
        try:
            return self._client.embed_documents(texts)
        except Exception as exc:
            logger.error(f"Gemini embed_documents hiba: {exc}")
            # A Google saját belső hibáját lefordítjuk a közös LLMServiceError típusunkra
            # (a "from exc" megőrzi az eredeti hiba nyomát is a konzolon hibakereséshez),
            # majd felküldjük a hívási lánc tetejére, hogy az API réteg egységesen kezelhesse.
            raise LLMServiceError(f"Gemini embedding hiba történt: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        # Egyetlen felhasználói kérdésből (text) készít vektort a kereséshez.
        try:
            return self._client.embed_query(text)
        except Exception as exc:
            logger.error(f"Gemini embed_query hiba: {exc}")
            raise LLMServiceError(f"Gemini lekérdezés embedding hiba történt: {exc}") from exc


class OllamaEmbedder(EmbeddingProvider):
    # Ugyanannak az EmbeddingProvider interfésznek az ingyenes, lokális (Ollama) megvalósítása.
    # Kívülről teljesen ugyanolyan metódusokat kínál, mint a GeminiEmbedder, ezért a fő program
    # kódmódosítás nélkül tud váltani a kettő között.

    def __init__(self, base_url: str, model_name: str) -> None:
        # Itt is csak a helyi Ollama szerver elérését állítjuk be, hívás még nincs.
        self._client = OllamaEmbeddings(
            base_url=base_url,
            model=model_name,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._client.embed_documents(texts)
        except Exception as exc:
            logger.error(f"Ollama embed_documents hiba: {exc}")
            raise LLMServiceError(f"Lokális Ollama embedding hiba (fut az Ollama szerver a {self._client.base_url} címen?): {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._client.embed_query(text)
        except Exception as exc:
            logger.error(f"Ollama embed_query hiba: {exc}")
            raise LLMServiceError(f"Lokális Ollama lekérdezés embedding hiba: {exc}") from exc