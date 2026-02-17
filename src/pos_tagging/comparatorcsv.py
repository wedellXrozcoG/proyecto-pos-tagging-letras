import pandas as pd
from nltk import pos_tag, word_tokenize
import spacy

# ES PARA CREAR EL CSV CON LA COMPARACIÓN
class POSComparator:
    def __init__(self, filepath):
        self.nlp = spacy.load("en_core_web_sm")
        self.corpus = pd.read_csv(filepath, sep=';')
        self.resultados = []

    def comparar_canciones(self, n_canciones=5):
        #Compara POS tagging entre NLTK y Spacy para n canciones
        for idx in range(n_canciones):
            comparison_sentence = self.corpus['text'].iloc[idx]

            # NLTK
            tokens = word_tokenize(comparison_sentence)
            nltk_tags = pos_tag(tokens)

            # Spacy
            doc = self.nlp(comparison_sentence)
            spacy_tags = [(token.text, token.pos_, token.tag_) for token in doc]

            # Guardar resultados
            for i, (word, nltk_tag) in enumerate(nltk_tags):
                if i < len(spacy_tags):
                    spacy_word, spacy_pos, spacy_fine = spacy_tags[i]
                    self.resultados.append({
                        'cancion_id': idx + 1,
                        'token': word,
                        'nltk_tag': nltk_tag,
                        'spacy_pos': spacy_pos,
                        'spacy_fine_pos': spacy_fine
                    })
                else:
                    self.resultados.append({
                        'cancion_id': idx + 1,
                        'token': word,
                        'nltk_tag': nltk_tag,
                        'spacy_pos': '',
                        'spacy_fine_pos': ''
                    })

            # Mostrar en consola
            print("\n" + "=" * 80)
            print(f"COMPARACIÓN: NLTK vs SPACY - Canción {idx + 1}")
            print("=" * 80)
            print(f"Texto: {comparison_sentence[:100]}...\n")
            print(f"{'Token':<20} {'NLTK Tag':<15} {'Spacy POS':<12} {'Spacy Fine POS':<15}")
            print("-" * 80)

            for i, (word, nltk_tag) in enumerate(nltk_tags):
                if i < len(spacy_tags):
                    spacy_word, spacy_pos, spacy_fine = spacy_tags[i]
                    print(f"{word:<20} {nltk_tag:<15} {spacy_pos:<12} {spacy_fine:<15}")
                else:
                    print(f"{word:<20} {nltk_tag:<15}")

    def guardar_csv(self, output_path="../data/results/comparisons.csv"):
        #Guarda los resultados en un CSV
        corpus_resultados = pd.DataFrame(self.resultados)
        corpus_resultados.to_csv(output_path, index=False, sep=';')
        print(f"\nResultados guardados en {output_path}")



comparator = POSComparator("../data/processed/spotify_clean02.csv")
comparator.comparar_canciones(n_canciones=5)
comparator.guardar_csv()