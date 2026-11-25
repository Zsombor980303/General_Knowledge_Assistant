# Saját hibatípusok közös deklarációja (olyan, mint C-ben a hibakódokat tartalmazó .h header fájl).
# Azért van külön fájlban, hogy az alsó rétegek (amelyek eldobják a hibát) és a felső API réteg
# (amelyik elkapja és HTTP hibakóddá alakítja) ugyanabból a közös forrásból importálják a neveket.
# A külső könyvtárak (Gemini, Ollama, ChromaDB) saját belső hibáit a kliensekben ezekre fordítjuk le.


class AppBaseError(Exception):
    # Közös ősosztály az összes saját alkalmazáshibának.
    # Ha ezzel kapunk el hibát, csak a saját, előre tervezett hibáinkat fogja meg;
    # a futásidejű kódolási bugokat (pl. KeyError) nem nyeli el, a program rendesen megáll.
    pass


class ConfigurationError(AppBaseError):
    # Hiányzó API-kulcs vagy érvénytelen .env beállítás esetén.
    pass


class VectorStoreError(AppBaseError):
    # A ChromaDB írása vagy olvasása közben fellépő hiba.
    pass


class LLMServiceError(AppBaseError):
    # Általános hiba a Gemini vagy Ollama hívásakor.
    # Mivel az LLMTimeoutError ebből származik, ez az általános LLM hibát és a timeoutot is elkapja.
    pass


class LLMTimeoutError(LLMServiceError):
    # Az LLM nem válaszolt a beállított időkorláton belül.
    # Külön el lehet kapni (pl. 504-es HTTP válaszhoz), vagy az LLMServiceError részeként.
    pass

class ValidationError(AppBaseError):
    """Érvénytelen bemeneti adat (pl. üres kérdés) esetén."""