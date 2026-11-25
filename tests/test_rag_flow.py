from src.graphs.workflow import NO_CONTEXT_ANSWER, build_rag_graph
from src.services.ingestion import split_text_into_chunks
from tests.conftest import FakeEmbedder, FakeLLM, FakeVectorStore


def test_split_text_into_chunks_creates_overlapping_pieces() -> None:
    # Ellenőrzi, hogy a szövegdaraboló a megadott méret és átfedés alapján bontja-e fel a szöveget.
    text = "A" * 1000
    chunks = split_text_into_chunks(text=text, source="test.md", chunk_size=500, chunk_overlap=100)

    assert len(chunks) == 3
    assert chunks[0].chunk_id == "test.md_0"
    assert chunks[0].source == "test.md"
    assert len(chunks[0].content) == 500


def test_rag_flow_generates_answer_when_relevant_chunks_exist(sample_retrieved_chunks) -> None:
    # Ellenőrzi a normál RAG ágat: ha vannak releváns találatok, meghívja az LLM-et,
    # a prompt tartalmazza a dokumentumok szövegét, és visszaadja a generált választ + forrásokat.
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(search_results=sample_retrieved_chunks)
    llm = FakeLLM(response_text="Heti 3 nap távmunka engedélyezett.")

    graph = build_rag_graph(embedder=embedder, vector_store=vector_store, llm=llm, top_k=3, min_score=0.25)
    result = graph.invoke({"question": "Hány nap home office jár?"})

    assert llm.was_called is True
    assert "Heti 3 nap home office" in llm.last_prompt
    assert result["answer"] == "Heti 3 nap távmunka engedélyezett."
    assert len(result["retrieved_chunks"]) == 2


def test_rag_flow_uses_fallback_without_calling_llm_when_no_chunks_found() -> None:
    # Ellenőrzi a fallback ágat: ha nincs találat az adatbázisban, az LLM meg sem hívódik,
    # és a biztonságos "nincs információ" választ adja vissza.
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(search_results=[])
    llm = FakeLLM(response_text="Ezt nem szabadna visszaadnia.")

    graph = build_rag_graph(embedder=embedder, vector_store=vector_store, llm=llm, top_k=3, min_score=0.25)
    result = graph.invoke({"question": "Mi a cég éves árbevétele?"})

    assert llm.was_called is False
    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["retrieved_chunks"] == []


def test_rag_flow_filters_out_low_score_chunks(sample_retrieved_chunks) -> None:
    # Ellenőrzi, hogy a min_score küszöb alatti találatokat a rendszer kiszűri és fallbacket ad.
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(search_results=sample_retrieved_chunks)
    llm = FakeLLM()

    graph = build_rag_graph(embedder=embedder, vector_store=vector_store, llm=llm, top_k=3, min_score=0.95)
    result = graph.invoke({"question": "Irreleváns kérdés"})

    assert llm.was_called is False
    assert result["answer"] == NO_CONTEXT_ANSWER