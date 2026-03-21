import pandas as pd

class SpotifyDataLoader:
    def __init__(self):
        self.corpus = pd.read_csv("../../data/raw/spotify_dataset.csv")


# Uso
loader = SpotifyDataLoader()
print(loader.corpus.head())