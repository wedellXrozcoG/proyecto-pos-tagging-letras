from loader import SpotifyDataLoader
import pandas as pd


class DataCleaner:
    def __init__(self):
        loader = SpotifyDataLoader()
        self.df = loader.df

        # Filtrar solo hip-hop
        self.df = self.df[self.df['Genre'] == 'hip hop']

        # Extraer solo el año de Release Date
        self.df['Release Date'] = self.df['Release Date'].str.extract(r'(\d{4})')

        self.df['Release Date'] = pd.to_numeric(self.df['Release Date'], errors='coerce')
        self.df = self.df[self.df['Release Date'] >= 1990]

        # Muestreo de 100,000 registros
        n_samples = min(len(self.df), 100000)
        if len(self.df) > n_samples:
            self.df = self.df.sample(n=n_samples, random_state=42)

        print("Columnas disponibles:", self.df.columns.tolist())

        # Usar solo columnas que creemos necesarias
        self.df = self.df[['text', 'emotion', 'Release Date', 'Genre', 'Explicit', 'Popularity']]

        self.df.to_csv("../../data/processed/spotify_clean.csv", index=False, sep=';')
        print(f"CSV guardado con {len(self.df)} filas")

cleaner = DataCleaner()