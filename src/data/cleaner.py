from loader import SpotifyDataLoader
import pandas as pd
import re

class DataCleaner:
    def __init__(self):
        loader = SpotifyDataLoader()
        self.df = loader.df

        # 1. Filtrar solo hip-hop
        self.df = self.df[self.df['Genre'] == 'hip hop']

        # 2. Extraer solo el año de Release Date
        self.df['Release Date'] = self.df['Release Date'].str.extract(r'(\d{4})')
        self.df['Release Date'] = pd.to_numeric(self.df['Release Date'], errors='coerce')
        self.df = self.df[self.df['Release Date'] >= 1990]

        # 3. Muestreo de registros
        n_samples = min(len(self.df), 100000)
        if len(self.df) > n_samples:
            self.df = self.df.sample(n=n_samples, random_state=42)

        # 4. Seleccionar columnas necesarias
        self.df = self.df[['text', 'emotion', 'Release Date', 'Genre', 'Explicit', 'Popularity']]

        # 5. Limpieza de texto (El método de abajo)
        print("Limpiando etiquetas de estructura [Intro, Chorus, etc.]...")
        self.df['text'] = self.df['text'].apply(self.limpiar_texto)

        # 6. Guardar CSV
        self.df.to_csv("../../data/processed/spotify_clean02.csv", index=False, sep=';')
        print(f"✅ CSV guardado con {len(self.df)} filas y texto limpio")


    def limpiar_texto(self, texto):
        if not isinstance(texto, str):
            return ""

        # Elimina corchetes y su contenido
        texto = re.sub(r'\[.*?\]', '', texto)

        # Elimina paréntesis (coros de fondo)
        texto = re.sub(r'\(.*?\)', '', texto)

        # Limpia saltos de línea y espacios múltiples
        texto = re.sub(r'\s+', ' ', texto)

        return texto.strip()

cleaner = DataCleaner()