from langgraph.graph import END, START, StateGraph

from src.core.logging import get_logger
from src.domain.interfaces import EmbeddingProvider, LLMProvider, VectorStoreProvider
from src.graphs.state import RAGState

logger = get_logger(__name__)

# Szigorú rendszerutasítás (system prompt) az AI-nak: a feladatleírás előírja, hogy
# a válasz kizárólag a visszakeresett dokumentumrészletekre támaszkodjon, és ne találjon ki adatokat.
SYSTEM_PROMPT = (
    "Te egy vállalati belső tudásbázis-asszisztens vagy. "
    "Válaszolj magyarul, tömören és pontosan, KIZÁRÓLAG az alábbi 'KONTEXTUS' részben szereplő információk alapján. "
    "Ha a kontextus nem tartalmazza a választ a kérdésre, írd le egyértelműen, hogy "
    "'A megadott belső tudásbázisban nem található releváns információ a kérdés megválaszolásához.' "
    "Soha ne használj külső tudást, és ne találj ki szabályokat."
)

NO_CONTEXT_ANSWER = "A megadott belső tudásbázisban nem található releváns információ a kérdés megválaszolásához."


# Összeállítja és lefordítja (compile) a LangGraph RAG munkafolyamatot.
# Paraméterek (itt kapja meg kívülről a kész "szerszámait" a main.py-ból):
# - embedder: A példányosított embedding szolgáltató (Gemini vagy Ollama) a kérdés vektorizálásához.
# - vector_store: A példányosított ChromaDB tároló a kereséshez.
# - llm: A példányosított válaszgeneráló AI modell (Gemini vagy Ollama).
# - top_k: Hány legjobban hasonlító dokumentumrészletet kérjen ki a kereső.
# - min_score: Minimális hasonlósági küszöb (0.0-1.0), ami alatt eldobjuk a találatot.
def build_rag_graph(
    embedder: EmbeddingProvider,
    vector_store: VectorStoreProvider,
    llm: LLMProvider,
    top_k: int = 3,
    min_score: float = 0.25,
):
    # A lenti belső függvényeket (retrieve_node, generate_node, stb.) nem mi hívjuk meg kézzel,
    # hanem átadjuk őket a gráfnak, és a LangGraph futtatáskor (.invoke()) magától lépteti végig őket.
    # Azért vannak ezen a függvényen belül, hogy közvetlenül elérjék a fenti paramétereket.

    # 1. ÁLLOMÁS: Ezt a függvényt a LangGraph motor hívja meg automatikusan, amikor a
    # qa_service.py-ban elindítjuk a gráfot a .invoke({"question": "..."}) paranccsal.
    # A kapott szótár lesz itt a 'state' bemeneti paraméter.
    def retrieve_node(state: RAGState) -> RAGState:
        # 1. ÁLLOMÁS: Kérdés vektorizálása és keresés a vektoros adatbázisban.
        question = state["question"]
        logger.info("LangGraph [retrieve]: releváns dokumentumrészletek keresése...")

        # A felhasználó szöveges kérdéséből számsort (vektort) csinálunk.
        query_embedding = embedder.embed_query(question)
        # A kereső visszatérési értéke (raw_chunks) már nem számsor, hanem a top_k darab
        # legjobban passzoló RetrievedChunk struct listája (szöveggel, forrással, pontszámmal).
        raw_chunks = vector_store.search(query_embedding=query_embedding, top_k=top_k)

        # Kiszűrjük a túl alacsony hasonlósági pontszámú (irreleváns) találatokat.
        relevant_chunks = [chunk for chunk in raw_chunks if chunk.score >= min_score]
        logger.info(
            f"LangGraph [retrieve]: {len(raw_chunks)} találatból {len(relevant_chunks)} érte el a minimum küszöböt ({min_score})."
        )

        # Csak a megváltozott mezőt adjuk vissza: a LangGraph automatikusan beleolvasztja a state-be.
        return {"retrieved_chunks": relevant_chunks}

    def generate_node(state: RAGState) -> RAGState:
        # 2. ÁLLOMÁS: Válasz generálása a megtalált dokumentumrészletek alapján.
        question = state["question"]
        chunks = state.get("retrieved_chunks", [])
        logger.info("LangGraph [generate]: válasz generálása az LLM segítségével...")

        # Összefűzzük a megtalált szövegrészleteket forrásmegjelöléssel, hogy az AI tudja, melyik fájlból dolgozik.
        context_blocks: list[str] = []
        for chunk in chunks:
            context_blocks.append(f"[Forrás: {chunk.source}]\n{chunk.content}")
        context_text = "\n\n---\n\n".join(context_blocks)

        user_prompt = (
            f"KONTEXTUS:\n{context_text}\n\n"
            f"FELHASZNÁLÓ KÉRDÉSE:\n{question}"
        )

        answer = llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        return {"answer": answer}

    def decide_after_retrieve(state: RAGState) -> str:
        # DÖNTÉSI PONT (vasúti váltó): A LangGraph szétválasztja a döntéshozót az állomásoktól.
        # Ez a függvény nem írhat bele a state-be, csak egy útvonalnevet ad vissza szövegként,
        # ami alapján a lenti path_map szótár eldönti, melyik állomásra ugorjon tovább a program.
        if not state.get("retrieved_chunks"):
            logger.warning("LangGraph [decision]: nincs releváns találat, LLM hívás kihagyva.")
            return "no_context"
        return "generate"

    def fallback_node(state: RAGState) -> RAGState:
        # Biztonsági mellékvágány: mivel a decide_after_retrieve nem módosíthatja a state-et,
        # ez a külön állomás írja bele a biztonsági üzenetet, ha nem volt releváns találat.
        return {"answer": NO_CONTEXT_ANSWER}

    # --- A GRÁF ÖSSZEÁLLÍTÁSA (ITT ALAKUL KI A VALÓDI FUTÁSI SORREND) ---
    # 1. Üres irányított gráf létrehozása a közös RAGState adatcsomaggal.
    graph = StateGraph(RAGState)

    # 2. Állomások (node) regisztrálása: névhez rendeljük a függvényeket (itt még nincs sorrend).
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    # 3. Nyilak (edge) behúzása = ez határozza meg a futási sorrendet:
    # 1. lépés: Mindig a kereséssel indulunk.
    graph.add_edge(START, "retrieve")

    # 2. lépés: A keresés után a decide_after_retrieve döntése alapján kettéválik az út.
    graph.add_conditional_edges(
        "retrieve",
        decide_after_retrieve,
        {
            "generate": "generate",   # Ha "generate" jött vissza -> AI válaszírás
            "no_context": "fallback", # Ha "no_context" jött vissza -> biztonsági üzenet
        },
    )

    # 3. lépés: Mindkét ág végén lezárjuk a folyamatot (END).
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    # 4. Ellenőrzi a tervrajzot (nincs-e zsákutca) és lefordítja egy futtatható objektummá.
    return graph.compile()