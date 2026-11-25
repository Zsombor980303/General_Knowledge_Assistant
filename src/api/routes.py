from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import AskRequest, AskResponse, ErrorResponse, SourceItem
from src.core.exceptions import AppBaseError, LLMServiceError, LLMTimeoutError, ValidationError, VectorStoreError
from src.core.logging import get_logger

logger = get_logger(__name__)

# Az APIRouter olyan, mint egy al-szerver, amire felregisztráljuk a végpontokat,
# és később a main.py-ban egyetlen sorral rácsatlakoztatjuk a fő FastAPI alkalmazásra.
router = APIRouter()

# A @router.get("/health") dekorátor a közvetlenül alatta lévő függvényt köti a GET /health útvonalhoz.
# Ez egy szabványos "életjel" (liveness) végpont: ha a szerver válaszol HTTP 200-zal, az bizonyítja,
# hogy az alkalmazás elindult, él és fogad kéréseket (ha összeomlott volna, nem jönne válasz).
# A request.app.state.settings pedig a fő FastAPI alkalmazás közös tárolórekeszéből veszi elő
# a main.py-ban elmentett .env beállításokat, hogy kiírhassuk az aktív AI szolgáltatók nevét.
@router.get("/health", summary="Alkalmazás állapotellenőrzés")
def health_check(request: Request) -> dict[str, str]:
    # Ellenőrzi, hogy a szerver fut-e, és visszaadja az aktív AI szolgáltató nevét.
    settings = request.app.state.settings
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    }


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    summary="Kérdés feltevése a belső tudásbázishoz",
)
def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    # A payload paramétert a FastAPI automatikusan beolvassa a bejövő JSON-ből és ellenőrzi az AskRequest alapján.
    # A request.app.state.qa_service pedig a main.py-ban példányosított QAService objektum.
    # Honnan tudja a FastAPI, hogy a bejövő JSON-t a payload változóba kell betöltenie?
    # A típusjelölésből (: AskRequest)! A FastAPI induláskor megvizsgálja a paramétereket:
    # - Ami Pydantic BaseModel típusú (AskRequest), azt automatikusan a bejövő HTTP kérés
    #   JSON törzséből (body) tölti fel (teljesen mindegy, mi a változó neve), így jön létre a payload.question.
    # - Ami Request típusú (request: Request), azt pedig a saját belső kérés-objektumából adja át,
    #   ahonnan a main.py-ban elmentett qa_service példányt vesszük elő (request.app.state.qa_service).
    qa_service = request.app.state.qa_service

    try:
        result = qa_service.ask(payload.question)

        # A belső RetrievedChunk struct-okat átmásoljuk a publikus JSON SourceItem formátumba.
        source_items: list[SourceItem] = []
        for chunk in result["sources"]:
            source_items.append(
                SourceItem(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    content=chunk.content,
                    score=chunk.score,
                )
            )

        return AskResponse(answer=result["answer"], sources=source_items)

    except ValidationError as exc:
        # Üres vagy hibás kérdés -> HTTP 400 Bad Request
        logger.warning(f"Validációs hiba a /ask végponton: {exc}")
        raise HTTPException(status_code=400, detail={"error": "validation_error", "detail": str(exc)}) from exc

    except LLMTimeoutError as exc:
        # Az AI modell nem válaszolt időben -> HTTP 504 Gateway Timeout
        logger.error(f"LLM időtúllépés a /ask végponton: {exc}")
        raise HTTPException(status_code=504, detail={"error": "llm_timeout", "detail": str(exc)}) from exc

    except LLMServiceError as exc:
        # Külső AI szolgáltató (Gemini / Ollama) hiba -> HTTP 502 Bad Gateway
        logger.error(f"LLM szolgáltatói hiba a /ask végponton: {exc}")
        raise HTTPException(status_code=502, detail={"error": "llm_service_error", "detail": str(exc)}) from exc

    except (VectorStoreError, AppBaseError) as exc:
        # Adatbázis vagy egyéb belső alkalmazáshiba -> HTTP 500 Internal Server Error
        logger.error(f"Belső alkalmazáshiba a /ask végponton: {exc}")
        raise HTTPException(status_code=500, detail={"error": "internal_error", "detail": str(exc)}) from exc