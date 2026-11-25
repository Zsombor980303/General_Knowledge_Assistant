import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.exceptions import VectorStoreError
from src.core.logging import get_logger
from src.domain.interfaces import VectorStoreProvider
from src.domain.models import DocumentChunk, RetrievedChunk

logger = get_logger(__name__)

# Az absztrakt VectorStoreProvider konkrét megvalósítása ChromaDB-vel.
# Fájlalapú (PersistentClient) módban használjuk, így a vektorok egy helyi mappába
# kerülnek a lemezre, nem kell hozzá külön adatbázis-szervert telepíteni.


class ChromaVectorStore(VectorStoreProvider):
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        # Példányosításkor itt csatlakoztatjuk az adatbázist és nyitjuk meg a táblát.
        try:
            # 1. Maga az adatbázis-motor elindítása: a PersistentClient a RAM helyett
            # a lemezre (a persist_dir mappába, pl. "chroma_db") menti az adatokat, így
            # újraindítás után is megmaradnak, nem kell újra vektorizálni a fájlokat.
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),  # Kikapcsolja a külső statisztikaküldést
            )
            # 2. Konkrét adattábla (collection) megnyitása vagy létrehozása az adatbázison belül. Egy adatbázison belül több gyűjteményünk is lehet (pl egy HR szabályzat, egy Általános tudásbázis, stb). Itt megadjuk, melyik gyűjtemény érdekel most minket.
            # (a collection_name pl. "company_knowledge_base"). A cosine metrika azt állítja be,
            # hogy a keresésnél koszinusz-hasonlóság alapján mérje a vektorok közelségét.
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.error(f"ChromaDB inicializálási hiba: {exc}")
            raise VectorStoreError(f"Nem sikerült megnyitni a vektoros tárolót: {exc}") from exc

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        # Elmenti a feldarabolt dokumentumokat (chunks = DocumentChunk lista) és a hozzájuk
        # tartozó vektorokat (embeddings = lista a listában, minden szövegrészlethez egy float lista).
        if not chunks:
            return
        try:
            # Az upsert (update + insert) azért jobb a sima add-nál, mert ha ugyanazzal az ID-val újra betöltjük a fájlokat, nem dob duplikációs hibát, hanem szépen frissíti őket.
            # A ChromaDB 4 párhuzamos listában várja az adatokat: nincs külön index-mező,
            # egyszerűen a listák sorrendje miatt tartozik össze az ids[0], documents[0],
            # embeddings[0] és metadatas[0] (mint 4 párhuzamos tömb C-ben).
            # A metadatas azért kell, mert a ChromaDB alapból csak a szöveget és a vektort tárolja:
            # itt mentjük el a forrásfájl nevét ("source"), hogy válaszadáskor vissza tudjuk adni a
            # forráshivatkozást is. A **chunk.metadata pedig egy bónusz kicsomagolás, hogy ha a
            # fájlnéven kívül mást is betettünk a chunk.metadata-ba (pl. chunk_index - azaz hányadik bekezdés az adott forrásban), az is mentődjön el.
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.content for chunk in chunks],
                embeddings=embeddings,
                metadatas=[{"source": chunk.source, **chunk.metadata} for chunk in chunks],
            )
        except Exception as exc:
            logger.error(f"ChromaDB mentési hiba: {exc}")
            raise VectorStoreError(f"Nem sikerült menteni a dokumentumrészleteket: {exc}") from exc

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        # Megkeresi a kérdés vektorához (amit az embed_query állított elő) legközelebb álló
        # top_k darab szövegrészletet, és saját RetrievedChunk listává alakítja őket.
        try:
            # Üres adatbázisra a ChromaDB hibát dobna, ezért ilyenkor azonnal üres listát adunk vissza.
            if self._collection.count() == 0:
                return []

            # A query_embeddings azért lista, mert a ChromaDB egyszerre több kérdésre is tud keresni.
            # Az n_results-nál a min() gondoskodik róla, hogy ne kérjünk több elemet, mint amennyi bent van.
            # Az include-ban megadjuk a kért kimeneti mezőket (az "ids"-t a ChromaDB mindig automatikusan visszaadja).
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            # A ChromaDB 2D-s listát (lista a listában) ad vissza, ahol a sorok a feltett kérdések.
            # Mivel mi egyszerre csak 1 kérdést küldünk be, a [0] indexszel kivesszük az első kérdés
            # összes találatát sima listákba. A [[]] biztonsági alapérték, ha üres lenne a válasz.
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            retrieved: list[RetrievedChunk] = []
            # A zip() elemről elemre összepárosítja a 4 listát (1. találat adatai, 2. találat adatai, stb.).
            for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
                # A ChromaDB távolságot ad vissza (0 = tökéletes egyezés, 1 = teljesen eltérő).
                # Kivonjuk 1-ből, hogy hasonlósági pontszámot kapjunk (minél nagyobb, annál jobb),
                # 4 tizedesjegyre kerekítve, és a max(0.0, ...) miatt sosem lesz negatív.
                # A for utáni 4 változónévbe (chunk_id, doc, meta, dist) a Python rögtön szét is osztja
                # az aktuális elemeket (ugyanaz, mint C-ben a for (i = 0; i < n; i++) négy tömbbel: ids[i], documents[i], stb.).
                similarity_score = round(max(0.0, 1.0 - float(dist)), 4)
                source = str(meta.get("source", "ismeretlen")) if meta else "ismeretlen"
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=str(chunk_id),
                        content=str(doc),
                        source=source,
                        score=similarity_score,
                    )
                )
            return retrieved
        except Exception as exc:
            logger.error(f"ChromaDB keresési hiba: {exc}")
            raise VectorStoreError(f"Hiba történt a vektoros keresés során: {exc}") from exc

    def count(self) -> int:
        # Visszaadja, hány darab dokumentumrészlet van jelenleg tárolva az adatbázisban.
        try:
            return self._collection.count()
        except Exception as exc:
            logger.error(f"ChromaDB darabszám lekérdezési hiba: {exc}")
            raise VectorStoreError(f"Nem sikerült lekérdezni a tárolt elemek számát: {exc}") from exc