from loader import SpotifyDataLoader
import pandas as pd
import re


class DataCleaner:
    def __init__(self):
        loader = SpotifyDataLoader()
        self.corpus = loader.corpus

        # 1. Ver cuántos registros hay de cada género
        print("📊 Registros por género:")
        print(f"Hip-hop: {len(self.corpus[self.corpus['Genre'] == 'hip hop']):,}")
        print(f"Pop: {len(self.corpus[self.corpus['Genre'] == 'pop']):,}\n")

        # 2. Filtrar hip-hop y pop
        self.corpus = self.corpus[self.corpus['Genre'].isin(['hip hop', 'pop'])]

        # 3. Extraer solo el año de Release Date
        self.corpus['Release Date'] = self.corpus['Release Date'].str.extract(r'(\d{4})')
        self.corpus['Release Date'] = pd.to_numeric(self.corpus['Release Date'], errors='coerce')
        self.corpus = self.corpus[self.corpus['Release Date'] >= 1990]

        # 4. Muestreo balanceado: 5000 hip-hop + 5000 pop
        hip_hop_corpus = self.corpus[self.corpus['Genre'] == 'hip hop']
        pop_corpus     = self.corpus[self.corpus['Genre'] == 'pop']

        hip_hop_sample = hip_hop_corpus.sample(n=min(len(hip_hop_corpus), 5000), random_state=42)
        pop_sample     = pop_corpus.sample(n=min(len(pop_corpus), 5000), random_state=42)

        self.corpus = pd.concat([hip_hop_sample, pop_sample], ignore_index=True)

        print(f"Muestra final:")
        print(f"Hip-hop: {len(hip_hop_sample):,}")
        print(f"Pop: {len(pop_sample):,}")
        print(f"Total: {len(self.corpus):,}\n")

        # 5. Seleccionar columnas necesarias — se agrega Artist(s) y song
        self.corpus = self.corpus[['song', 'Artist(s)', 'text', 'emotion', 'Release Date', 'Genre', 'Explicit', 'Popularity']]

        # 6. Renombrar columnas
        self.corpus = self.corpus.rename(columns={'Artist(s)': 'artist'})

        # 7. Reemplazar nulos con "Desconocido"
        self.corpus = self.corpus.fillna('Desconocido')

        # 8. Limpieza de texto
        print("Limpiando texto...")
        self.corpus['text'] = self.corpus['text'].apply(self.limpiar_texto)

        # 9. Eliminar filas con texto vacío o nulo después de limpiar
        self.corpus = self.corpus[self.corpus['text'].notna() & (self.corpus['text'].str.len() > 0)]

        # 10. Guardar CSV
        self.corpus.to_csv("../../data/processed/spotify_clean02.csv", index=False, sep=';')
        print(f"✅ CSV guardado con {len(self.corpus)} filas y texto limpio")

    def limpiar_texto(self, texto):
        if not isinstance(texto, str):
            return ""

        # Elimina corchetes y su contenido [Intro, Chorus, etc.]
        texto = re.sub(r'\[.*?\]', '', texto)

        # Elimina paréntesis (coros de fondo)
        texto = re.sub(r'\(.*?\)', '', texto)

        # Limpia caracteres raros - deja solo letras, números, espacios y apóstrofes
        texto = re.sub(r'[^\w\s\']', ' ', texto)

        # Limpia saltos de línea y espacios múltiples
        texto = re.sub(r'\s+', ' ', texto)

        return texto.strip()


cleaner = DataCleaner()