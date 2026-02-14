import pandas as pd

class SpotifyDataLoader:
    def __init__(self):
        self.df = pd.read_csv("../../data/raw/spotify_dataset.csv")


# Uso
loader = SpotifyDataLoader()
print(loader.df.head())