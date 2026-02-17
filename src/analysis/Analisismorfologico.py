import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import spacy


class AnalisisMorfologico:
    def __init__(self, filepath):
        self.nlp = spacy.load("en_core_web_sm")
        self.corpus = pd.read_csv(filepath, sep=';')
        self.resultados = []

    def procesar_corpus(self):
        """Procesa el corpus completo con spaCy"""
        print("Procesando corpus...")

        for idx, row in self.corpus.iterrows():
            doc = self.nlp(row['text'][:512])

            self.resultados.append({
                'genero': row['Genre'],
                'año': row['Release Date'],
                'pos_tags': [t.pos_ for t in doc if not t.is_punct and not t.is_space],
                'fine_tags': [t.tag_ for t in doc if not t.is_punct and not t.is_space],
                'pronombres': [t.text.lower() for t in doc if t.pos_ == 'PRON'],
                'tokens': [t for t in doc if not t.is_punct and not t.is_space]
            })

            if (idx + 1) % 500 == 0:
                print(f"  ✓ {idx + 1}/{len(self.corpus)} procesadas")

        print("✅ Procesamiento completo\n")

    # ============================================
    # 1. DISTRIBUCIÓN POS COMPLETA (6 pts)
    # ============================================
    def distribucion_pos_completa(self):
        """Muestra distribución completa de todas las categorías POS"""
        print("=" * 60)
        print("1. DISTRIBUCIÓN POS COMPLETA")
        print("=" * 60 + "\n")

        all_pos = [pos for r in self.resultados for pos in r['pos_tags']]
        pos_counter = Counter(all_pos)

        # Tabla completa
        print(f"{'Categoría POS':<15} {'Frecuencia':<12} {'Porcentaje':<12}")
        print("-" * 60)
        for pos, count in pos_counter.most_common():
            porcentaje = (count / len(all_pos)) * 100
            print(f"{pos:<15} {count:<12,} {porcentaje:<12.2f}%")

        # Gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        names = list(pos_counter.keys())
        values = list(pos_counter.values())
        ax.bar(names, values, color=plt.cm.tab20(range(len(names))), edgecolor='black')
        ax.set_xlabel('Categoría POS', fontweight='bold')
        ax.set_ylabel('Frecuencia', fontweight='bold')
        ax.set_title('Distribución Completa de Categorías POS', fontweight='bold', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        print()

    # ============================================
    # 2. MÉTRICAS DERIVADAS (6 pts)
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
    # 3. ANÁLISIS DE PRONOMBRES (6 pts)
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
    # 4. PATRONES ESPECÍFICOS DEL GÉNERO (6 pts)
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

    # ============================================
    # 5. INTERPRETACIÓN CONTEXTUALIZADA (6 pts)
    # ============================================
    def interpretacion_contextualizada(self):
        """Interpreta los hallazgos morfológicos en contexto"""
        print("=" * 60)
        print("5. INTERPRETACIÓN CONTEXTUALIZADA")
        print("=" * 60 + "\n")

        all_pos = [pos for r in self.resultados for pos in r['pos_tags']]
        pos_counter = Counter(all_pos)

        total = len(all_pos)
        sustantivos = sum(1 for pos in all_pos if pos == 'NOUN')
        verbos = sum(1 for pos in all_pos if pos == 'VERB')
        pronombres = sum(1 for pos in all_pos if pos == 'PRON')

        print("📝 INTERPRETACIÓN DE HALLAZGOS:\n")

        print("1️⃣ Densidad de Sustantivos:")
        sust_pct = sustantivos / total * 100
        print(f"   → {sust_pct:.2f}% del texto son sustantivos")
        if sust_pct > 25:
            print("   → Alta densidad: Indica narrativas descriptivas y concretas")
        else:
            print("   → Baja densidad: Lenguaje más abstracto y conceptual")

        print(f"\n2️⃣ Uso de Verbos:")
        verb_pct = verbos / total * 100
        print(f"   → {verb_pct:.2f}% del texto son verbos")
        if verb_pct > 15:
            print("   → Alto dinamismo: Énfasis en acciones y movimiento")
        else:
            print("   → Bajo dinamismo: Enfoque en estados y descripciones")

        print(f"\n3️⃣ Pronombres:")
        pron_pct = pronombres / total * 100
        print(f"   → {pron_pct:.2f}% del texto son pronombres")

        # Análisis por género
        pronombres_hiphop = [p for r in self.resultados if r['genero'] == 'hip hop' for p in r['pronombres']]
        pronombres_pop = [p for r in self.resultados if r['genero'] == 'pop' for p in r['pronombres']]

        i_hiphop = sum(1 for p in pronombres_hiphop if p == 'i')
        i_pop = sum(1 for p in pronombres_pop if p == 'i')

        print(f"\n4️⃣ Diferencias entre géneros:")
        print(f"   Hip-Hop: {i_hiphop:,} usos de 'I' → Narrativa autobiográfica fuerte")
        print(f"   Pop: {i_pop:,} usos de 'I' → {'Mayor' if i_pop > i_hiphop else 'Menor'} énfasis en lo personal")

        print(f"\n5️⃣ Conclusión General:")
        print(f"   → El corpus muestra un lenguaje {'lírico-narrativo' if sust_pct > 25 else 'conceptual-abstracto'}")
        print(f"   → {'Alta' if verb_pct > 15 else 'Baja'} orientación hacia la acción")
        print(f"   → Perspectiva {'fuertemente' if pron_pct > 10 else 'moderadamente'} centrada en lo personal")
        print()

    def ejecutar_analisis_completo(self):
        """Ejecuta todos los análisis morfológicos"""
        self.procesar_corpus()
        self.distribucion_pos_completa()
        self.calcular_metricas_derivadas()
        self.analisis_pronombres()
        self.patrones_por_genero()
        self.interpretacion_contextualizada()


# Uso
analisis = AnalisisMorfologico("../../data/processed/spotify_clean02.csv")
analisis.ejecutar_analisis_completo()