import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import spacy


class AnalisisMorfologico:
    def __init__(self, dataframe):
        self.nlp = spacy.load("en_core_web_sm")
        self.corpus = dataframe
        self.resultados = []

    def procesar_corpus(self):
        """Procesa el corpus completo con spaCy"""
        print("Procesando corpus...")

        for idx, row in self.corpus.iterrows():
            doc = self.nlp(row['text'][:512]) #Toma solo los primeros 512 caracteres de la letra

                #Diccionario
            self.resultados.append({
                'genero': row['Genre'],
                'año': row['Release Date'],
                'pos_tags': [t.pos_ for t in doc if not t.is_punct and not t.is_space], #Exceptúa espacios y puntuaciones
                'fine_tags': [t.tag_ for t in doc if not t.is_punct and not t.is_space],
                'pronombres': [t.text.lower() for t in doc if t.pos_ == 'PRON'],
                'tokens': [t for t in doc if not t.is_punct and not t.is_space]
            })

            if (idx + 1) % 500 == 0:
                print(f"  ✓ {idx + 1}/{len(self.corpus)} procesadas")

        print("✅ Procesamiento completo\n")

    # ============================================
    # 1. DISTRIBUCIÓN POS COMPLETA
    # ============================================
    def distribucion_pos_completa(self, genero_objetivo=None):
        """Muestra distribución completa. Si genero_objetivo es None, muestra todo."""
        from collections import Counter
        import matplotlib.pyplot as plt

        # 1. Filtrar los datos por género
        if genero_objetivo:
            datos_filtrados = [r for r in self.resultados if r['genero'] == genero_objetivo]
            titulo_extra = f" (Género: {genero_objetivo})"
        else:
            datos_filtrados = self.resultados
            titulo_extra = " (Todos los géneros)"

        if not datos_filtrados:
            print(f"No se encontraron datos para el género: {genero_objetivo}")
            return

        # 2. Aplanar la lista de pos_tags
        all_pos = [pos for r in datos_filtrados for pos in r['pos_tags']]
        total_tokens = len(all_pos)
        pos_counter = Counter(all_pos)

        # --- IMPRESIÓN DE TABLA ---
        print("=" * 60)
        print(f"1. DISTRIBUCIÓN POS COMPLETA{titulo_extra}")
        print("=" * 60 + "\n")
        print(f"{'Categoría POS':<15} {'Frecuencia':<12} {'Porcentaje':<12}")
        print("-" * 60)

        for pos, count in pos_counter.most_common():
            porcentaje = (count / total_tokens) * 100
            print(f"{pos:<15} {count:<12,} {porcentaje:<12.2f}%")

        # --- GRÁFICO ---
        # Ordenar por frecuencia para que el gráfico
        datos_ordenados = pos_counter.most_common()
        names = [x[0] for x in datos_ordenados]
        values = [x[1] for x in datos_ordenados]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(names, values, color=plt.cm.viridis(range(len(names))), edgecolor='black')

        ax.set_xlabel('Categoría POS', fontweight='bold')
        ax.set_ylabel('Frecuencia', fontweight='bold')
        ax.set_title(f'Distribución de Categorías POS{titulo_extra}', fontweight='bold', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Añadir el número encima de cada barra
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{int(yval):,}', va='bottom', ha='center', fontsize=9)

        plt.tight_layout()
        plt.show()

    # ============================================
    # 2. MÉTRICAS DERIVADAS
    # ============================================
    def calcular_metricas_derivadas(self):
        """Calcula métricas morfológicas avanzadas"""
        print("=" * 60)
        print("2. MÉTRICAS DERIVADAS")
        print("=" * 60 + "\n")

        all_pos = [pos for r in self.resultados for pos in r['pos_tags']]

        # Agrupar por categorías principales
        sustantivos = sum(1 for pos in all_pos if pos.startswith('NOUN'))
        verbos = sum(1 for pos in all_pos if pos.startswith('VERB'))
        adjetivos = sum(1 for pos in all_pos if pos.startswith('ADJ'))
        adverbios = sum(1 for pos in all_pos if pos.startswith('ADV'))
        pronombres = sum(1 for pos in all_pos if pos == 'PRON')
        determinantes = sum(1 for pos in all_pos if pos == 'DET')

        total = len(all_pos)

        print("📊 MÉTRICAS BÁSICAS:")
        print(f"  Total de tokens: {total:,}")
        print(f"  Sustantivos: {sustantivos:,} ({sustantivos / total * 100:.2f}%)")
        print(f"  Verbos: {verbos:,} ({verbos / total * 100:.2f}%)")
        print(f"  Adjetivos: {adjetivos:,} ({adjetivos / total * 100:.2f}%)")
        print(f"  Adverbios: {adverbios:,} ({adverbios / total * 100:.2f}%)")
        print(f"  Pronombres: {pronombres:,} ({pronombres / total * 100:.2f}%)")

        # Ratios derivados
        print(f"\n📈 RATIOS DERIVADOS:")
        print(f"  Ratio Sustantivo/Verbo: {sustantivos / verbos:.2f}")
        print(f"  Ratio Adjetivo/Sustantivo: {adjetivos / sustantivos:.2f}")
        print(f"  Densidad léxica: {(sustantivos + verbos + adjetivos + adverbios) / total * 100:.2f}%")
        print(f"  Complejidad sintáctica: {verbos / total * 100:.2f}%")

        # Gráfico comparativo
        categorias = ['Sustantivos', 'Verbos', 'Adjetivos', 'Adverbios', 'Pronombres', 'Determinantes']
        valores = [sustantivos, verbos, adjetivos, adverbios, pronombres, determinantes]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(categorias, valores, color=plt.cm.Set3(range(6)), edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Frecuencia', fontweight='bold')
        ax.set_title('Comparación de Categorías Principales', fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        print()

    # ============================================
    # 3. ANÁLISIS DE PRONOMBRES
    # ============================================
    def analisis_pronombres(self):
        """Análisis detallado del uso de pronombres"""
        print("=" * 60)
        print("3. ANÁLISIS DE PRONOMBRES")
        print("=" * 60 + "\n")

        # Pronombres por género
        pronombres_hiphop = [p for r in self.resultados if r['genero'] == 'hip hop' for p in r['pronombres']]
        pronombres_pop = [p for r in self.resultados if r['genero'] == 'pop' for p in r['pronombres']]

        counter_hiphop = Counter(pronombres_hiphop)
        counter_pop = Counter(pronombres_pop)

        print("🎤 HIP-HOP - Top 10 Pronombres:")
        for pron, count in counter_hiphop.most_common(10):
            porcentaje = (count / len(pronombres_hiphop)) * 100
            print(f"  {pron:<10} → {count:,} ({porcentaje:.2f}%)")

        print("\n🎵 POP - Top 10 Pronombres:")
        for pron, count in counter_pop.most_common(10):
            porcentaje = (count / len(pronombres_pop)) * 100
            print(f"  {pron:<10} → {count:,} ({porcentaje:.2f}%)")

        # Clasificación por persona
        primera_persona = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        segunda_persona = ['you', 'your', 'yours']
        tercera_persona = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']

        p1_hiphop = sum(counter_hiphop[p] for p in primera_persona)
        p2_hiphop = sum(counter_hiphop[p] for p in segunda_persona)
        p3_hiphop = sum(counter_hiphop[p] for p in tercera_persona)

        p1_pop = sum(counter_pop[p] for p in primera_persona)
        p2_pop = sum(counter_pop[p] for p in segunda_persona)
        p3_pop = sum(counter_pop[p] for p in tercera_persona)

        print(f"\n👤 DISTRIBUCIÓN POR PERSONA:")
        print(f"  Hip-Hop: 1ª persona {p1_hiphop:,}, 2ª persona {p2_hiphop:,}, 3ª persona {p3_hiphop:,}")
        print(f"  Pop:     1ª persona {p1_pop:,}, 2ª persona {p2_pop:,}, 3ª persona {p3_pop:,}")

        # Gráfico comparativo
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        personas = ['1ª Persona', '2ª Persona', '3ª Persona']
        hiphop_vals = [p1_hiphop, p2_hiphop, p3_hiphop]
        pop_vals = [p1_pop, p2_pop, p3_pop]

        ax1.bar(personas, hiphop_vals, color=['#e74c3c', '#3498db', '#2ecc71'], edgecolor='black')
        ax1.set_title('Pronombres por Persona - Hip-Hop', fontweight='bold')
        ax1.set_ylabel('Frecuencia')

        ax2.bar(personas, pop_vals, color=['#e74c3c', '#3498db', '#2ecc71'], edgecolor='black')
        ax2.set_title('Pronombres por Persona - Pop', fontweight='bold')
        ax2.set_ylabel('Frecuencia')

        plt.tight_layout()
        plt.show()
        print()

    # ============================================
    # 4. PATRONES ESPECÍFICOS DEL GÉNERO
    # ============================================
    def patrones_por_genero(self):
        """Identifica patrones morfológicos únicos de cada género"""
        print("=" * 60)
        print("4. PATRONES ESPECÍFICOS POR GÉNERO")
        print("=" * 60 + "\n")

        # POS tags por género
        pos_hiphop = [pos for r in self.resultados if r['genero'] == 'hip hop' for pos in r['pos_tags']]
        pos_pop = [pos for r in self.resultados if r['genero'] == 'pop' for pos in r['pos_tags']]

        counter_hh = Counter(pos_hiphop)
        counter_pop = Counter(pos_pop)

        total_hh = len(pos_hiphop)
        total_pop = len(pos_pop)

        # Calcular diferencias relativas
        diferencias = {}
        for pos in set(list(counter_hh.keys()) + list(counter_pop.keys())):
            freq_hh = (counter_hh[pos] / total_hh * 100) if total_hh > 0 else 0
            freq_pop = (counter_pop[pos] / total_pop * 100) if total_pop > 0 else 0
            diferencias[pos] = freq_hh - freq_pop

        # Patrones distintivos
        print("🎤 PATRONES DISTINTIVOS DE HIP-HOP:")
        for pos, diff in sorted(diferencias.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {pos:<10} → +{diff:.2f}% más frecuente que en Pop")

        print("\n🎵 PATRONES DISTINTIVOS DE POP:")
        for pos, diff in sorted(diferencias.items(), key=lambda x: x[1])[:5]:
            print(f"  {pos:<10} → +{abs(diff):.2f}% más frecuente que en Hip-Hop")

        # Gráfico comparativo
        top_pos = [item[0] for item in counter_hh.most_common(10)]
        hh_vals = [(counter_hh[pos] / total_hh * 100) for pos in top_pos]
        pop_vals = [(counter_pop[pos] / total_pop * 100) for pos in top_pos]

        x = range(len(top_pos))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar([i - width / 2 for i in x], hh_vals, width, label='Hip-Hop', color='#e74c3c', edgecolor='black')
        ax.bar([i + width / 2 for i in x], pop_vals, width, label='Pop', color='#3498db', edgecolor='black')

        ax.set_xlabel('Categoría POS', fontweight='bold')
        ax.set_ylabel('Porcentaje (%)', fontweight='bold')
        ax.set_title('Comparación de Patrones POS: Hip-Hop vs Pop', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(top_pos, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        print()

