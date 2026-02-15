#%% md
# ## Instalación de dependencias
#%%
#!pip install -q nltk pandas matplotlib seaborn

#%% md
# ## Importación de librerías necesarias
#%%
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
#%%
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, pos_tag_sents
from nltk.corpus import wordnet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("✓ Librerías importadas correctamente")
print("✓ Configuración SSL aplicada")
#%% md
# ## Recursos necesarios de NLTK
#%%
print("Cargando recursos de NLTK...\n")

try:
    nltk.data.find('tokenizers/punkt_tab') # en inglés / como se está usando una diferente vesrión de NLTK entonces se cambia a punk_tab
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng') # inglés
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

print("✓ Recursos de NLTK listos")

print("\n" + "="*60)
print("¡Listo para comenzar con el POS Tagging!")
print("="*60)
#%% md
# ## POS Tagging NLTK con Spotify_clean02.csv
#%%
# carga de spotify_clean02.csv
df = pd.read_csv("../data/processed/spotify_clean02.csv", sep=';')
#%% md
# ## Primero una prueba con una fila
#%%
# Tomar una canción de ejemplo (la primera fila)
sentence = df['text'].iloc[0]

print("ORIGINAL:")
print(f'"{sentence}"\n')

# Paso 1: Tokenizar la oración
tokens = word_tokenize(sentence)
print("TOKENS:")
print(tokens)
print()

# Paso 2: Realizar POS tagging con NLTK
pos_tags_nltk = pos_tag(tokens)
print("POS TAGGING CON NLTK:")
for word, tag in pos_tags_nltk:
    print(f"{word:15} → {tag}")
#%% md
# ## Limpieza / Tokenización / POS Tagging con el df completo
#%%
# Múltiples oraciones con NLTK - Dataset completo
print("\n" + "="*60)
print("ANALIZANDO MÚLTIPLES CANCIONES CON NLTK")
print("="*60 + "\n")

# Procesar todas las canciones (o las primeras N si quieres ver ejemplos)
n_canciones = len(df)  # Cambia a 10 si solo quieres ver ejemplos

for idx in range(n_canciones):
    sentence = df['text'].iloc[idx]
    emotion = df['emotion'].iloc[idx]

    print(f"Canción {idx + 1} - Emoción: {emotion}")
    print(f"Texto: {sentence[:100]}...")  # Muestra solo los primeros 100 caracteres

    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    for word, tag in pos_tags:
        print(f"  {word:15} → {tag}")
    print()

    # Mostrar solo las primeras 5 canciones como ejemplo
    if idx == 4:
        print(f"\n... (procesando las {n_canciones - 5} canciones restantes en segundo plano)")
        break

print(f"\n✓ Total procesadas: {n_canciones} canciones")
#%% md
# ## Gráfico / Análisis Morfológico
#%%
# Procesar y recolectar POS tags
print("Procesando canciones...")
all_pos_tags = []
puntuacion = {'.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '-', '--', '...', '`', '``', "''"}

for idx in range(len(df)):
    tokens = word_tokenize(df['text'].iloc[idx])
    pos_tags = pos_tag(tokens)
    all_pos_tags.extend([tag for word, tag in pos_tags if word not in puntuacion])

print(f"✓ {len(df)} canciones procesadas\n")

# Top 10 y gráfico
pos_counts = Counter(all_pos_tags)
top_10 = pos_counts.most_common(10)
names = [t[0] for t in top_10]
values = [t[1] for t in top_10]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(names, values, color=plt.cm.Set3(range(10)), edgecolor='black', linewidth=1.5)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('POS Tag', fontsize=12, fontweight='bold')
ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
ax.set_title('Top 10 POS Tags - Hip-Hop', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()
#%%
print(f" Total palabras: {len(all_pos_tags):,}")
print(f" Tipos diferentes de POS tags: {len(pos_counts)}\n")

print("TOP 10 POS tags más frecuentes:")
for tag, count in top_10:
    percentage = (count / len(all_pos_tags)) * 100
    print(f"  {tag:10} → {count:,} ({percentage:.2f}%)")

# Análisis por categoría gramatical principal
sustantivos = sum([count for tag, count in pos_counts.items() if tag.startswith('NN')])
verbos = sum([count for tag, count in pos_counts.items() if tag.startswith('VB')])
adjetivos = sum([count for tag, count in pos_counts.items() if tag.startswith('JJ')])
adverbios = sum([count for tag, count in pos_counts.items() if tag.startswith('RB')])

print(f"\n Distribución por categoría:")
print(f"  Sustantivos: {sustantivos:,} ({sustantivos/len(all_pos_tags)*100:.2f}%)")
print(f"  Verbos:      {verbos:,} ({verbos/len(all_pos_tags)*100:.2f}%)")
print(f"  Adjetivos:   {adjetivos:,} ({adjetivos/len(all_pos_tags)*100:.2f}%)")
print(f"  Adverbios:   {adverbios:,} ({adverbios/len(all_pos_tags)*100:.2f}%)")