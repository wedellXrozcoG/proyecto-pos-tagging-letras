# Documentación del uso del IA

## 1.Herramientas Utilizadas:
Gemini(Google):Gemini (Google): Utilizado como asistente para documentación, ayuda en codigo, consultas sobre warnings y errores además de consultas/pensamientos.

Claude: Utilizado como asistente para ayuda en codigo y consultas/pensamientos.


## 2. Ejemplos de Prompts Representativos
- Prompt 1: Con spacy como puedo calcular "x", "y", "z".

- Prompt 2: En esta linea de codigo "codigo x" esta función "Funcion x"¿Para que me sirve?¿Podria quitarla?

- Prompt 3: Como puedo optimizar mi codigo "codigo_x" y puedo agregarle "x_funcion"

- Prompt 4: Necesito hacer un gráfico que me ayude a hace runa comparación entre 'x' y 'y', sobre etiquetas, cómo se puede hacer?

- Prompt 5: Cómo hago para meter este tag X en el botón de Y

- Prompt 6: Acabo de detectar un problema: cada vez que quiero devolverme al tag (ya en la página) se tienen que cargar muchos d registros de las décadas a cada rato y dura un montón de tiempo, eso no me sirve, puedes decirme cómo implementar una solución?

- Prompt 7: Cómo puedo implementar un diseño sobrio en Dash, que contenga como base la estructura y llamados que ya tengo?: (código)

## 3.Reflexión sobre cómo la IA ayudó en su aprendizaje
Reflexión 1: De mi parte me ayudo bastante en el proyecto en general pero formo parte fundamental en el desarrollo del Analisis mofologico me ayudo en codigo, errores, en optimización.

Reflexión 2: En mi caso tuvo una gran aportación. Principalmente cuando se necesitaba gráficos de lo que necesitaba y pudiera funcionar en el Dash. Además, para el diseño del dashboard me ayudó con la creación del css
y darle un diseño sobrio para hacerle diseño al machote que ya tenía (botones, tags, callbacks) con ayuda de la documentación en la página de plotly.
También, para la limpieza y filtros de x cosas del corpus. 

Ayudó MUCHÍSIMO en la optimización general del dashboard, ya que, anteriormente cuando tenía la clase del análisis ejecutando, si quería devolverme al análisis temporal por décadas
duraba bastante en volver a cargar el gráfico (porque tenía que hacer el análisis en todos los años de nuevo) por lo que, me recomendó utilizar caché y de esa manera mejorar el rendimiento.

## 4.Qué modificaciones hicieron al código/análisis generado por IA
- Mejoras en la visualización de resultados: Se optimizaron los print y formatos de salida para facilitar la lectura e interpretación de los resultados.
- Se quitaban líneas de código, se adaptaba a lo que teníamos porque traía cosas innecesarias (como librerías).
- Muchas veces inventaba rutas a otros archivos o no recordaba que algo ya estaba entonces se corregía.