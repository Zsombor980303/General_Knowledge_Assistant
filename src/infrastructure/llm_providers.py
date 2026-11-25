from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from src.core.exceptions import ConfigurationError, LLMServiceError, LLMTimeoutError
from src.core.logging import get_logger
from src.domain.interfaces import LLMProvider

logger = get_logger(__name__)

# Ez a fájl a válaszgeneráló nyelvi modelleket (LLM) valósítja meg az LLMProvider interfész alapján.
# Nem öröklődik az embedders.py-ból: az embedding csak kereséshez csinál számsorokat a szövegből,
# itt viszont már maga a chat modell fogalmazza meg a választ a megtalált dokumentumrészletekből.


def _is_timeout_error(exc: Exception) -> bool:
    # Belső segédfüggvény (az aláhúzás jelzi, hogy csak ebben a fájlban használjuk).
    # Mivel a Google és az Ollama más-más szöveggel dob időtúllépést (pl. "Deadline Exceeded" vs "timed out"),
    # itt egységesen eldöntjük, hogy timeout történt-e, hogy külön 504-es hibára tudjuk fordítani.
    error_text = str(exc).lower()
    return (isinstance(exc, TimeoutError) or "timeout" in error_text or "timed out" in error_text or "deadline" in error_text)


class GeminiLLM(LLMProvider):
    # Az absztrakt LLMProvider konkrét, felhős (Google Gemini) megvalósítása.
    # Itt még csak az osztályt definiáljuk, a tényleges példányosítás a main.py-ban történik induláskor.

    def __init__(self, api_key: str, model_name: str, timeout_seconds: float) -> None:
        # Példányosításkor itt még csak beállítjuk a klienst (kulcs, modell, időkorlát),
        # tényleges hálózati kérés csak a generate() metódus meghívásakor indul.
        if not api_key:
            raise ConfigurationError("A Gemini LLM használatához kötelező a GEMINI_API_KEY megadása.")
        self._client = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            timeout=timeout_seconds,
            temperature=0.0,  # 0.0 = tényalapú válasz, minimalizálja a hallucinációt
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Összeállítjuk a rendszerutasítást (viselkedési szabályok) és a kontextussal dúsított kérdést.
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        try:
            # A LangChain egységes felületén minden modellnek .invoke() a hívó függvénye
            # (szemben a nyers Google SDK .create() hívásával), így a Gemini és az Ollama ugyanúgy hívható.
            response = self._client.invoke(messages)
            return str(response.content)
        except Exception as exc:
            logger.error(f"Gemini LLM hívási hiba: {exc}")
            if _is_timeout_error(exc):
                raise LLMTimeoutError("A Gemini modell nem válaszolt a beállított időkorláton belül.") from exc
            raise LLMServiceError(f"Gemini modell hiba történt: {exc}") from exc


class OllamaLLM(LLMProvider):
    # Ugyanannak az LLMProvider interfésznek az ingyenes, lokális (Ollama) megvalósítása.
    # Mivel ugyanazt a generate() metódust kínálja, mint a GeminiLLM, a RAG-folyamat
    # kódmódosítás nélkül tud váltani a kettő között.

    def __init__(self, base_url: str, model_name: str, timeout_seconds: float) -> None:
        self._base_url = base_url
        self._client = ChatOllama(
            base_url=base_url,
            model=model_name,
            client_kwargs={"timeout": timeout_seconds},
            temperature=0.0,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        try:
            response = self._client.invoke(messages)
            return str(response.content)
        except Exception as exc:
            logger.error(f"Ollama LLM hívási hiba: {exc}")
            if _is_timeout_error(exc):
                raise LLMTimeoutError("A lokális Ollama modell nem válaszolt időben.") from exc
            raise LLMServiceError(
                f"Lokális Ollama hiba (fut az Ollama szerver a {self._base_url} címen?): {exc}"
            ) from exc