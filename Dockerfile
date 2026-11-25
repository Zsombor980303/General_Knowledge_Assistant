FROM python:3.11-slim

# Ne keletkezzenek .pyc fájlok a konténerben, és a Python logok pufferelés nélkül, azonnal jelenjenek meg a konzolon.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Munkakönyvtár létrehozása és belépés az /app mappába.
# Így a projektfájlok nem a konténer gyökerébe (/) szóródnak szét, és minden további művelet innen indul.
WORKDIR /app

# Először csak a requirements.txt másolása és telepítése történik: így kódmódosításkor
# a Docker gyorsítótárból (cache) veszi elő a csomagokat, és nem telepíti újra őket minden buildnél.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# A forráskód és a tudásbázis dokumentumok bemásolása a konténerbe.
COPY src ./src
COPY data ./data

# A ChromaDB lokális adatbázis könyvtárának létrehozása.
RUN mkdir -p /app/chroma_db

# Dokumentációs jelzés arról, hogy a konténerben futó alkalmazás a 8000-es hálózati porton figyel.
# (A tényleges porttovábbítást a host gép felé a docker-compose.yml "8000:8000" beállítása végzi.)
EXPOSE 8000

# A konténer indító parancsa, amely automatikusan lefut a konténer indulásakor:
# - "uvicorn": a Python webszerver-motor
# - "src.main:app": a src/main.py fájlból tölti be az 'app' nevű FastAPI példányt
# - "--host 0.0.0.0": minden hálózati interfészen figyel (kötelező a konténeren kívüli eléréshez)
# - "--port 8000": a 8000-es porton fogadja a bejövő HTTP kéréseket
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]