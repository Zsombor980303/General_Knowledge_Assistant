from fastapi import FastAPI

from src.api.routes import router
from src.core.config import Settings, get_settings
from src.core.logging import get_logger, setup_logging
from src.graphs.workflow import build_rag_graph
from src.infrastructure.embedders import GeminiEmbedder, OllamaEmbedder
from src.infrastructure.llm_providers import GeminiLLM, OllamaLLM
from src.infrastructure.vector_store import ChromaVectorStore
from src.services.ingestion import ingest_documents
from src.services.qa_service import QAService

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    # Ez az alkalmazás "main" vezénylőfüggvénye: fentről lefelé összedugja az összes réteget.
    active_settings = settings or get_settings()
    setup_logging(active_settings.log_level)

    logger.info("Alkalmazás inicializálása...")

    # 1. Embedding driver kiválasztása a .env beállítás alapján (Gemini vagy lokális Ollama).
    if active_settings.embedding_provider == "ollama":
        embedder = OllamaEmbedder(
            base_url=active_settings.ollama_base_url,
            model_name=active_settings.ollama_embedding_model,
        )
    else:
        embedder = GeminiEmbedder(
            api_key=active_settings.gemini_api_key,
            model_name=active_settings.gemini_embedding_model,
        )

    # 2. Vektoros adatbázis (ChromaDB) példányosítása.
    vector_store = ChromaVectorStore(
        persist_dir=active_settings.chroma_persist_dir,
        collection_name=active_settings.chroma_collection_name,
    )

    # 3. Válaszgeneráló AI (LLM) driver kiválasztása a .env beállítás alapján.
    if active_settings.llm_provider == "ollama":
        llm = OllamaLLM(
            base_url=active_settings.ollama_base_url,
            model_name=active_settings.ollama_llm_model,
            timeout_seconds=active_settings.llm_timeout_seconds,
        )
    else:
        llm = GeminiLLM(
            api_key=active_settings.gemini_api_key,
            model_name=active_settings.gemini_llm_model,
            timeout_seconds=active_settings.llm_timeout_seconds,
        )

    # 4. Tudásbázis dokumentumok (.md, .txt) beolvasása, darabolása és mentése a ChromaDB-be. A loaded_chunks-ban megkapjuk, hogy hány dokumentum részlet került éppen elmentésre.
    loaded_chunks = ingest_documents(
        data_dir=active_settings.data_dir,
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=active_settings.chunk_size,
        chunk_overlap=active_settings.chunk_overlap,
    )
    logger.info(f"Induláskori tudásbázis betöltés kész ({loaded_chunks} darab).")

    # 5. LangGraph munkafolyamat és a QAService összerakása.
    rag_graph = build_rag_graph(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        top_k=active_settings.retrieval_top_k,
        min_score=active_settings.min_score,
    )
    qa_service = QAService(rag_graph=rag_graph)

    # 6. FastAPI webszerver létrehozása, a kész objektumok elmentése az app.state tárolóba
    # (pont innen veszi elő őket a routes.py!), és a végpontok rácsatlakoztatása.
    app = FastAPI(
        title="Enterprise General Knowledge Assistant",
        description="Vállalati belső tudásbázis-asszisztens RAG és LangGraph alapokon.",
        version="1.0.0",
    )
    # Betesszük a szerver közös "tárolórekeszébe" (app.state) a kész beállításokat és a qa_service
    # példányt, hogy a routes.py végpontjai elérjék őket (request.app.state...) globális változók nélkül.
    app.state.settings = active_settings
    app.state.qa_service = qa_service
    # Rácsatlakoztatjuk a routes.py-ban definiált végpontokat (GET /health, POST /ask) a fő szerverre.
    app.include_router(router)

    return app


# Ezt az 'app' példányt indítja el a webszerver (uvicorn src.main:app --reload).
app = create_app()