import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
            doc = self.nlp(row['text'][:512])  # Toma solo los primeros 512 caracteres de la letra

            # Diccionario
            self.resultados.append({
                'genero': row['Genre'],
                'año': row['Release Date'],
                'pos_tags': [t.pos_ for t in doc if not t.is_punct and not t.is_space],
                # Exceptúa espacios y puntuaciones
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
        # 1. Filtrar datos
        datos = [r for r in self.resultados if r['genero'] == genero_objetivo] if genero_objetivo else self.resultados
        titulo = f"Género: {genero_objetivo}" if genero_objetivo else "Todos los géneros"

        # 2. Contar etiquetas
        all_pos = [pos for r in datos for pos in r['pos_tags']]
        pos_counter = Counter(all_pos)
        total = len(all_pos)

        # 3. Mostrar Tabla
        print(f"\n--- DISTRIBUCIÓN POS: {titulo} ---")
        for pos, count in pos_counter.most_common():
            print(f"{pos:<10} | {count:>10,} | {count / total:>6.2%}")

        # 4. Gráfico con Plotly
        ordenado = pos_counter.most_common()
        df_pos = pd.DataFrame(ordenado, columns=['POS', 'Frecuencia'])

        fig = px.bar(df_pos, x='POS', y='Frecuencia',
                     title=f"Frecuencia POS - {titulo}")
        fig.update_traces(marker_color='skyblue', marker_line_color='black', marker_line_width=1.5)
        fig.update_layout(xaxis_title='POS', yaxis_title='Frecuencia')

        return fig

    # ============================================
    # 2. MÉTRICAS DERIVADAS
    # ============================================
    def calcular_metricas_derivadas(self, genero_objetivo=None):
        from collections import Counter

        # 1. Filtrar datos por género (si no se pasa género, usa todos)
        datos = [r for r in self.resultados if not genero_objetivo or r['genero'] == genero_objetivo]
        titulo = f"GÉNERO: {genero_objetivo.upper()}" if genero_objetivo else "TODOS LOS GÉNEROS"

        if not datos:
            print(f"No hay datos para el género: {genero_objetivo}")
            return

        # 2. Contar todas las etiquetas de una vez
        all_pos = [p for r in datos for p in r['pos_tags']]
        c = Counter(all_pos)
        total = len(all_pos)

        # 3. Agrupar categorías principales
        sust = c['NOUN'] + c['PROPN']
        verb = c['VERB'] + c['AUX']
        adj = c['ADJ']
        adv = c['ADV']
        pron = c['PRON']
        det = c['DET']

        print("=" * 60)
        print(f"MÉTRICAS: {titulo}")
        print("=" * 60)

        print(f"MÉTRICAS BÁSICAS:")
        print(f"  Total de tokens: {total:,}")
        print(f"  Sustantivos:  {sust:,} ({sust / total:.2%})")
        print(f"  Verbos:       {verb:,} ({verb / total:.2%})")
        print(f"  Adjetivos:    {adj:,} ({adj / total:.2%})")
        print(f"  Adverbios:    {adv:,} ({adv / total:.2%})")
        print(f"  Pronombres:   {pron:,} ({pron / total:.2%})")

        # 4. Ratios derivados (max(1, x) evita errores de división por cero)
        print(f"\nRATIOS DERIVADOS:")
        print(f"  Ratio Sustantivo/Verbo:    {sust / max(1, verb):.2f}")
        print(f"  Ratio Adjetivo/Sustantivo: {adj / max(1, sust):.2f}")
        print(f"  Densidad léxica:           {(sust + verb + adj + adv) / total:.2%}")
        print(f"  Complejidad sintáctica:    {verb / total:.2%}")

        # 5. Gráfico con Plotly
        cats = ['Sustantivos', 'Verbos', 'Adjetivos', 'Adverbios', 'Pronombres', 'Determinantes']
        vals = [sust, verb, adj, adv, pron, det]

        df_metricas = pd.DataFrame({'Categoría': cats, 'Frecuencia': vals})

        fig = px.bar(df_metricas, y='Categoría', x='Frecuencia',
                     orientation='h',
                     title=f'Distribución Morfológica - {titulo}',
                     color='Frecuencia',
                     color_continuous_scale='rainbow')
        fig.update_traces(marker_line_color='black', marker_line_width=1.5)
        fig.update_layout(xaxis_title='Frecuencia', yaxis_title='Categoría')

        return fig

    # ============================================
    # 3. ANÁLISIS DE PRONOMBRES
    # ============================================
    def analisis_pronombres(self, gen1="hip hop", gen2="pop"):
        # Listas de personas gramaticales
        p1_list = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        p2_list = ['you', 'your', 'yours']
        p3_list = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']

        def obtener_datos(genero):
            pronombres = [p.lower() for r in self.resultados if r['genero'] == genero for p in r['pronombres']]
            conteo = Counter(pronombres)
            # Sumar por persona
            p1 = sum(conteo[p] for p in p1_list)
            p2 = sum(conteo[p] for p in p2_list)
            p3 = sum(conteo[p] for p in p3_list)
            return pronombres, conteo, [p1, p2, p3]

        # Obtener datos para ambos géneros
        prons1, counts1, pers1 = obtener_datos(gen1)
        prons2, counts2, pers2 = obtener_datos(gen2)

        # --- IMPRESIÓN DE RESULTADOS ---
        print(f"{gen1.upper()} - Top 10 Pronombres:")
        for pron, cant in counts1.most_common(10):
            print(f"  {pron:<10} → {cant:,} ({cant / len(prons1):.2%})")

        print(f"\n{gen2.upper()} - Top 10 Pronombres:")
        for pron, cant in counts2.most_common(10):
            print(f"  {pron:<10} → {cant:,} ({cant / len(prons2):.2%})")

        print(f"\nDISTRIBUCIÓN POR PERSONA:")
        print(f"  {gen1.capitalize():<10}: 1ª pers {pers1[0]:,}, 2ª pers {pers1[1]:,}, 3ª pers {pers1[2]:,}")
        print(f"  {gen2.capitalize():<10}: 1ª pers {pers2[0]:,}, 2ª pers {pers2[1]:,}, 3ª pers {pers2[2]:,}")

        # --- GRÁFICO CON PLOTLY ---
        df_pron = pd.DataFrame({
            'Persona': ['1ª Persona', '2ª Persona', '3ª Persona'] * 2,
            'Frecuencia': pers1 + pers2,
            'Género': [gen1.upper()] * 3 + [gen2.upper()] * 3
        })

        fig = px.bar(df_pron, x='Persona', y='Frecuencia',
                     color='Género', barmode='group',
                     title=f"Comparación de Pronombres: {gen1.upper()} vs {gen2.upper()}",
                     color_discrete_map={gen1.upper(): '#e74c3c', gen2.upper(): '#3498db'})
        fig.update_traces(marker_line_color='black', marker_line_width=1.5)

        return fig

    # ============================================
    # 4. PATRONES ESPECÍFICOS DEL GÉNERO
    # ============================================
    def patrones_por_genero(self, gen1="hip hop", gen2="pop"):
        # 1. Obtener etiquetas POS y totales por género
        pos1 = [pos for r in self.resultados if r['genero'] == gen1 for pos in r['pos_tags']]
        pos2 = [pos for r in self.resultados if r['genero'] == gen2 for pos in r['pos_tags']]

        c1, t1 = Counter(pos1), len(pos1)
        c2, t2 = Counter(pos2), len(pos2)

        # 2. Seleccionar las 10 categorías más comunes del primer género para comparar
        top_pos = [item[0] for item in c1.most_common(10)]

        # 3. Convertir a porcentajes para que la comparación sea justa
        vals1 = [(c1[p] / t1 * 100) if t1 > 0 else 0 for p in top_pos]
        vals2 = [(c2[p] / t2 * 100) if t2 > 0 else 0 for p in top_pos]

        # 4. Crear Gráfico con Plotly
        df_patrones = pd.DataFrame({
            'POS': top_pos * 2,
            'Porcentaje': vals1 + vals2,
            'Género': [gen1.upper()] * 10 + [gen2.upper()] * 10
        })

        fig = px.bar(df_patrones, x='POS', y='Porcentaje',
                     color='Género', barmode='group',
                     title=f'Patrones POS: {gen1.upper()} vs {gen2.upper()}',
                     color_discrete_map={gen1.upper(): '#e74c3c', gen2.upper(): '#3498db'})
        fig.update_traces(marker_line_color='black', marker_line_width=1.5)
        fig.update_layout(yaxis_title='Porcentaje (%)', xaxis_title='POS')

        return fig