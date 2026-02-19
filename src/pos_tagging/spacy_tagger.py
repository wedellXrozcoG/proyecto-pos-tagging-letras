import time  # Importante para medir el tiempo
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

# Importacion de archivo preprocessorEDA.py
ruta_EDA = r"D:\proyecto-pos-tagging-letras\src\data"
for r in [ruta_EDA]:
    if r not in sys.path:
        sys.path.append(r)
from preprocessorEDA import preprocesador

#          ***********************************Ejecución****************************************

class pos_spacy(preprocesador):
    def __init__(self, df):
        super().__init__(df)
        self.nlp = spacy.load("en_core_web_sm")

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


    def multiples_spacyF2(self, columna, batch_size: int = 2000):
        resultados = []

        textos = self.df[columna].astype(str)

        # tqdm envuelve al generador y muestra el progreso
        generador = nlp.pipe(textos, batch_size=batch_size)

        #Usamos len(textos) que ahora ya existe
        for doc in tqdm(generador, total=len(textos), desc="Procesando"):
            tokens_del_texto = []

            for token in doc:
                tokens_del_texto.append(
                    (token.text, token.pos_, token.tag_)
             )

            resultados.append(tokens_del_texto)

        return resultados

    def auto_grafico_pos_total(self, col_txt, col_exp):
        print(f"📊 Generando gráfico total basado en '{col_exp}'...")

        # 1. Expandir la columna 'pos_tags'
        # Cada fila ahora será un token individual
        df_tokens = self.df.explode('pos_tags')

        # 2. Extraer el TAG
        df_tokens['tag_especifico'] = df_tokens['pos_tags'].apply(
            lambda x: x[2] if isinstance(x, list) or isinstance(x, tuple) else None
        )

        # 3. Limpiar nulos y filtrar el Top 5
        df_tokens = df_tokens.dropna(subset=['tag_especifico'])
        top_5_idx = df_tokens['tag_especifico'].value_counts().head(5).index
        df_filtrado = df_tokens[df_tokens['tag_especifico'].isin(top_5_idx)]

        # 4. Crear tabla cruzada para el apilado
        tabla = pd.crosstab(df_filtrado['tag_especifico'], df_filtrado[col_exp]).loc[top_5_idx]
        tabla.columns = ['No Explícita (0)', 'Explícita (1)']

        # 5. Configuración del gráfico
        plt.style.use('dark_background')
        ax = tabla.plot(kind='bar', stacked=True, figsize=(12, 7),
                        color=['#4a6d6a', '#7d4a4a'], edgecolor='white')

        # Añadir los números totales sobre/dentro de las barras
        for rect in ax.patches:
            height = rect.get_height()
            if height > 0:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + height / 2,
                        f'{int(height):,}',
                        ha='center', va='center', color='white',
                        fontweight='bold', fontsize=9)

        plt.title(f'Análisis Morfológico Total: Top 5 Tags vs {col_exp}', fontsize=14)
        plt.ylabel('Cantidad Total de Apariciones')
        plt.xlabel('Etiqueta Gramatical (POS Tag)')
        plt.xticks(rotation=0)
        plt.legend(title='Estado Binario')
        plt.tight_layout()
        plt.show()

    def multiples_spacyF2_time(self, columna, batch_size: int = 2000):
        resultados = []
        textos = self.df[columna].astype(str)
        total_canciones = len(textos)

        # 1. Capturamos el tiempo de inicio
        inicio = time.time()

        # Procesamiento con el generador de spaCy
        generador = nlp.pipe(textos, batch_size=batch_size)

        for doc in tqdm(generador, total=total_canciones, desc="Procesando"):
            tokens_del_texto = []
            for token in doc:
                tokens_del_texto.append(
                    (token.text, token.pos_, token.tag_)
                )
            resultados.append(tokens_del_texto)

        # 2. Capturamos el tiempo final y calculamos
        fin = time.time()
        segundos_totales = fin - inicio

        # --- CÁLCULOS DE RENDIMIENTO ---
        minutos = int(segundos_totales // 60)
        segundos_restantes = segundos_totales % 60

        # Formula Canciones por segundo
        velocidad = total_canciones / segundos_totales

        # Formula Latencia
        latencia = (segundos_totales / total_canciones) * 1000

        # 3. Imprimimos el reporte de rendimiento
        print(f"\n¡Proceso finalizado!")
        print(f"--- MÉTRICAS DE RENDIMIENTO ---")
        print(f"Tiempo total: {minutos} min {segundos_restantes:.2f} seg")
        print(f"Velocidad: {velocidad:.2f} canciones/seg") # Velocidad = Numero total de canciones/segundos totales
        print(f"Latencia: {latencia:.2f} ms por canción") # Latencia = (segundos totales / total de canciones)* 1000 milisegundos

        return

    def pipeline_completo(self, columna, max_registros=None):
        """Procesa el pipeline completo de spaCy."""
        # Limitar registros
        textos = self.df[columna].head(max_registros).astype(str).tolist()  # ← self.df

        # Procesar
        filas = []
        for cancion_id, doc in enumerate(tqdm(self.nlp.pipe(textos), total=len(textos)), start=1):  # ← self.nlp
            for token in doc:
                filas.append({
                    'cancion_id': cancion_id,
                    'token': token.text,  # 1. TOKENIZACIÓN
                    'pos': token.pos_,    # 2. ETIQUETADO POS
                    'tag': token.tag_,    #    ETIQUETADO POS (fino)
                    'is_stopword': token.is_stop, # 3. STOPWORDS
                    'entity': token.ent_type_,    # 4. NER
                    'lower': token.lower_, # 5. Mayúsculas/minúsculas
                    'lemma': token.lemma_, # 6. LEMATIZACIÓN
                })

        return pd.DataFrame(filas)
