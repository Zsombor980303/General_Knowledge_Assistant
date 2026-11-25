from pathlib import Path

from src.core.logging import get_logger
from src.domain.interfaces import EmbeddingProvider, VectorStoreProvider
from src.domain.models import DocumentChunk

logger = get_logger(__name__)

# Támogatott dokumentumformátumok (a feladatleírás által kért Markdown és sima szövegfájlok).
SUPPORTED_EXTENSIONS = {".md", ".txt"}


def split_text_into_chunks(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    # Egy hosszú dokumentumszöveget kisebb, átfedő szövegrészletekre (DocumentChunk) vág.
    # Az átfedés (chunk_overlap) azért fontos, hogy egy szabály vagy mondat ne vágódjon ketté
    # értelmetlenül a darabolási határon.
    # A külső .strip() levágja a fájl elejéről és végéről a felesleges üres sorokat/szóközöket.
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0
    text_length = len(cleaned_text)

    # Lépésköz: ennyivel ugrunk előre minden körben (pl. 500 - 100 = 400 karakterrel).
    # A max(1, ...) védelem: ha valaki hibásan nagyobbra állítaná az átfedést, mint a darabméret,
    # akkor is előrelép legalább 1 karaktert, így nem ragad végtelen ciklusba.
    step = max(1, chunk_size - chunk_overlap)

    while start < text_length:
        end = start + chunk_size
        # A cleaned_text[start:end] kivágja a szöveget start-tól end-ig (slicing).
        # A belső .strip() azért kell ide is, mert a vágási ponton (szöveg közepén) is kerülhet
        # sortörés vagy felesleges szóköz az új darab elejére vagy végére.
        piece = cleaned_text[start:end].strip()

        if piece:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source}_{chunk_index}",
                    content=piece,
                    source=source,
                    metadata={"chunk_index": chunk_index},
                )
            )
            chunk_index += 1

        start += step

    # A visszatérő lista DocumentChunk struct-okat tartalmaz (a memóriában csak a 4 mező van).
    return chunks


def ingest_documents(
    data_dir: str,
    embedder: EmbeddingProvider,
    vector_store: VectorStoreProvider,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> int:
    # A "munkavezető" függvény: beolvassa a fájlokat a lemezről, meghívja rájuk a fenti
    # darabolót (split_text_into_chunks), vektorrá alakítja, és elmenti a ChromaDB-be.
    directory = Path(data_dir)
    if not directory.exists() or not directory.is_dir():
        logger.warning(f"A megadott dokumentum mappa nem létezik: {data_dir}")
        return 0

    all_chunks: list[DocumentChunk] = []

    # directory.iterdir(): kilistázza a mappa összes elemét.
    # sorted(...): ABC sorrendbe rendezi a fájlokat a kiszámítható betöltésért.
    for file_path in sorted(directory.iterdir()):
        # is_file(): tényleg fájl-e (nem almappa).
        # suffix.lower(): lekéri a kiterjesztést (pl. ".MD"), kisbetűsíti, és megnézi, támogatott-e.
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            # read_text(): egyetlen paranccsal megnyitja, beolvassa egy stringbe, és le is zárja a fájlt.
            content = file_path.read_text(encoding="utf-8")
            file_chunks = split_text_into_chunks(
                text=content,
                source=file_path.name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(file_chunks)
            logger.info(f"Dokumentum feldarabolva: {file_path.name} -> {len(file_chunks)} részlet.")

    if not all_chunks:
        logger.warning("Nem található feldolgozható (.md vagy .txt) dokumentum a tudásbázis mappában.")
        return 0

    # 1. Kigyűjtjük külön csak a nyers szövegeket egy listába, mert az embedder csak string listát vár.
    texts: list[str] = []
    for chunk in all_chunks:
        texts.append(chunk.content)

    # 2. Az összes szövegrészletből egyszerre számsorokat (vektorokat) gyártunk (list[list[float]]).
    embeddings = embedder.embed_documents(texts)

    # 3. Elmentjük a szövegdarabokat (metaadatokkal) és a hozzájuk tartozó vektorokat a ChromaDB-be.
    vector_store.add_chunks(chunks=all_chunks, embeddings=embeddings)

    logger.info(f"Tudásbázis betöltés kész: összesen {len(all_chunks)} részlet elmentve.")
    return len(all_chunks)