from tqdm import tqdm
# Descargar modelo de Spacy
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"])

# Importar todas las librerías necesarias
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Cargar modelo de Spacy en inglés
print("Cargando modelo de Spacy...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ Modelo de Spacy cargado correctamente")
except OSError:
    print("⚠ Modelo no encontrado. Instalando...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")
    print("✓ Modelo de Spacy instalado y cargado")

print("\n" + "="*60)
print("¡Listo para comenzar con el POS Tagging!")
print("="*60)

#          ***********************************Ejecución****************************************

class pos_spacy:
    def __init__(self, df):
        self.df = df


    def unica_spacy(self, columna):
        # Procesar la oración con Spacy
        sentence_spacy = self.df[columna].sample(1).iloc[0]
        doc = nlp(sentence_spacy)

        print("ORACIÓN ORIGINAL:")
        print(f'"{sentence_spacy}"\n')

        print("POS TAGGING CON SPACY(Una sola canción):")
        print("Token              POS    Fine POS  Lemma")
        print("-" * 50)
        for token in doc:
            print(f"{token.text:15}  {token.pos_:6}  {token.tag_:8}  {token.lemma_}")

    def multiples_spacy(self, columna, mostrar_solo=3, n_sample=1000):
        # Múltiples oraciones con Spacy
        print("\n" + "=" * 60)
        print(f"ANALIZANDO UNA MUESTRA DE {n_sample} CANCIONES CON SPACY")
        print(f"(Mostrando detalle de las primeras {mostrar_solo})")
        print("=" * 60 + "\n")

        # CREAR EL SAMPLE: Elegimos n_sample filas al azar
        # Usamos random_state para que si lo corres de nuevo, salgan las mismas canciones
        dataset_reducido = self.df[columna].sample(n=n_sample, random_state=42)

        for i, sent in enumerate(dataset_reducido):

            # Procesamos con el modelo optimizado
            with nlp.select_pipes(disable=["ner", "parser"]):
                doc = nlp(str(sent))

            if i < mostrar_solo:
                # Mostramos un fragmento de la oración
                print(f"Oración #{i + 1} (Sample): {str(sent)[:100]}...")
                for token in doc:
                    print(f"  {token.text:15} → {token.pos_} ({token.tag_})")
                print("-" * 30)

            elif i % 100 == 0:  # Aviso más frecuente ya que el sample es corto
                print(f"... Procesando canción {i} de {n_sample} (Muestra) ...")

        print("\n" + "=" * 60)
        print("¡MUESTRA PROCESADA EXITOSAMENTE!")
        print("=" * 60)


    def pos_tag_con_progreso(self, columna, batch_size: int = 2000):
        resultados = []

        # CORRECCIÓN: Definimos 'textos' para que len(textos) no de error
        textos = self.df[columna].astype(str)

        # tqdm envuelve al generador y muestra el progreso
        # CORRECCIÓN: Pasamos 'textos' al generador
        generador = nlp.pipe(textos, batch_size=batch_size)

        # CORRECCIÓN: Usamos len(textos) que ahora ya existe
        for doc in tqdm(generador, total=len(textos), desc="Procesando"):
            tokens_del_texto = []

            for token in doc:
                tokens_del_texto.append(
                    (token.text, token.pos_, token.tag_)
             )

            resultados.append(tokens_del_texto)

        return resultados

    def auto_grafico_pos(self, col_txt, col_exp, n=1000):
        # 1. Procesamiento compacto con nlp.pipe
        sample = self.df.sample(min(n, len(self.df)))

        # Extraemos etiquetas ignorando basura en una sola línea
        tags = []
        for doc, exp in zip(nlp.pipe(sample[col_txt].astype(str)), sample[col_exp]):
            tags.extend([{'POS': t.tag_, 'Explicit': 'Sí' if exp == 1 else 'No'}
                            for t in doc if not t.is_punct and not t.is_space])

            # 2. Preparación rápida
            df_plot = pd.DataFrame(tags)
            top_5 = df_plot['POS'].value_counts().head(5).index
            df_plot = df_plot[df_plot['POS'].isin(top_5)]

            # 3. Gráfico directo
            plt.style.use('dark_background')
            plt.figure(figsize=(10, 5))

            ax = sns.countplot(data=df_plot, x='POS', hue='Explicit',
                               order=top_5, palette=['#7d4a4a', '#4a6d6a'])

            # Etiquetas de datos (Análisis visual rápido)
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                            ha='center', va='bottom', color='white', fontweight='bold', xytext=(0, 3),
                            textcoords='offset points')

            plt.title('Top 5 Funciones Gramaticales')
            plt.show()


