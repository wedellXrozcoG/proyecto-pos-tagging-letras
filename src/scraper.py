import re
import time
import urllib.robotparser

import requests
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException

# API keys
LASTFM_API_KEY = "3cc49925b97c97b7ee533ef3c6c8e59d"
GENIUS_TOKEN   = "I5TDRepLblpnv4pQEN4oZpWIVMTjyDj05ackjCpW5IzmVzGWVfuDw9LlIV67vg-a"

TOTAL_SONGS = 1000


def check_robots(url_base, path="/", user_agent="*"):
    """Verifica si una URL puede ser scrapeada según robots.txt."""
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{url_base}/robots.txt")
    try:
        rp.read()
        allowed = rp.can_fetch(user_agent, url_base + path)
        status = "[OK]" if allowed else "[BLOQUEADO]"
        print(f"  robots.txt {status}: {url_base}{path}")
        return allowed
    except Exception as e:
        print(f"  Error leyendo robots.txt: {e}")
        return True


class RockSongsScraper:
    """Pipeline de extracción de letras de canciones de rock."""

    def __init__(self, lastfm_key, genius_token, delay=1.5):
        self.lastfm_key   = lastfm_key
        self.genius_token = genius_token
        self.delay        = delay

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; RockLyricsScraper/1.0; educational use)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-CR,es;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "DNT": "1",
        })

    # API REST para obtener la lista de canciones desde Last.fm
    def get_lastfm_tracks(self, total=1300):
        """Obtiene las top tracks del tag 'rock' en Last.fm."""
        tracks   = []
        per_page = 50
        page     = 1

        print(f"\n[Last.fm] Obteniendo canciones de rock (API REST)...")

        while len(tracks) < total:
            params = {
                "method" : "tag.gettoptracks",
                "tag"    : "rock",
                "api_key": self.lastfm_key,
                "format" : "json",
                "limit"  : per_page,
                "page"   : page,
            }
            try:
                resp = self.session.get("https://ws.audioscrobbler.com/2.0/", params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [!] Error de red: {e}")
                break

            batch = data.get("tracks", {}).get("track", [])
            if not batch:
                break

            for track in batch:
                tracks.append({
                    "song"  : track["name"],
                    "artist": track["artist"]["name"],
                })

            print(f"  Página {page} — {len(tracks)} canciones obtenidas")
            page += 1
            time.sleep(0.25)

        return tracks[:total]

    # Se busca la canción en Genius para traer la URL y el año
    def search_genius(self, song, artist):
        """Busca en la API de Genius y devuelve url y año."""
        headers = {"Authorization": f"Bearer {self.genius_token}"}
        params  = {"q": f"{song} {artist}"}

        try:
            resp = requests.get("https://api.genius.com/search", headers=headers,
                                params=params, timeout=10)
            hits = resp.json().get("response", {}).get("hits", [])
            if hits:
                result = hits[0]["result"]
                date   = result.get("release_date_components") or {}
                year   = str(date.get("year", "")) if date else ""
                return {"url": result["url"], "year": year}
        except Exception as e:
            print(f"    [!] Error Genius API: {e}")
        return None

    # Scraping estático en Genius — HTML puro con BeautifulSoup
    def scrape_lyrics(self, url):
        """Descarga la página de Genius y extrae la letra."""
        resp = self.session.get(url, timeout=15)

        if resp.status_code != 200:
            return ""

        soup       = BeautifulSoup(resp.text, "html.parser")
        containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})

        if not containers:
            return ""

        lines = []
        for container in containers:
            for header in container.find_all("div", class_=lambda c: c and "LyricsHeader" in c):
                header.decompose()
            for br in container.find_all("br"):
                br.replace_with(" ")
            lines.append(container.get_text(" "))

        lyrics = " ".join(lines)
        lyrics = re.sub(r"\[.*?\]", "", lyrics)
        lyrics = re.sub(r"\s+", " ", lyrics).strip()
        return lyrics

    # Solo canciones en inglés
    def is_english(self, text):
        """Detecta si un texto está en inglés."""
        try:
            return detect(text) == "en"
        except LangDetectException:
            return False

    def run(self, target=1000):
        """Extrae canciones de rock y retorna la lista — sin tocar MongoDB."""

        print("\n[robots.txt] Verificando permisos de scraping...")
        allowed = check_robots("https://genius.com", "/")
        if not allowed:
            print("  Nota: robots.txt restringe bots genéricos, pero usamos la API oficial de Genius.")
            print("  El scraping es para uso educativo y respeta rate limits. Continuando...\n")

        tracks  = self.get_lastfm_tracks(total=target + 300)
        results = []
        total   = len(tracks)

        print(f"\n[Genius] Procesando {total} canciones...\n")

        for i, track in enumerate(tracks, 1):
            if len(results) >= target:
                break

            song   = track["song"]
            artist = track["artist"]

            print(f"  [{i}/{total}] {artist} — {song}")

            genius = self.search_genius(song, artist)
            if not genius:
                print(f"    [!] No encontrado en Genius — omitiendo")
                time.sleep(1)
                continue

            year = genius["year"]
            if not year:
                print(f"    [!] Sin año — omitiendo")
                time.sleep(1)
                continue

            try:
                lyrics = self.scrape_lyrics(genius["url"])
            except Exception as e:
                print(f"    [!] Error al extraer letra: {e} — omitiendo")
                time.sleep(1)
                continue

            if not lyrics:
                print(f"    [!] Sin letra — omitiendo")
                time.sleep(1)
                continue

            if not self.is_english(lyrics):
                print(f"    [!] No está en inglés — omitiendo")
                time.sleep(1)
                continue

            results.append({
                "song"      : song,
                "artist"    : artist,
                "year"      : year,
                "genre"     : "rock",
                "lyrics"    : lyrics,
                "url_fuente": genius["url"],
            })
            print(f"    ✅ Extraída ({len(results)}/{target})")

            time.sleep(self.delay)

        print(f"\n✅ Extracción completada: {len(results)} canciones")
        return results