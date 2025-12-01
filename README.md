# Implementación de técnicas de aprendizaje máquina

#### Angel Mauricio Ramírez Herrera | A01710158

## Resumen

Se desarrollaron dos modelos de clasificación que su objetivo principal es categorizar letras de canciones según su género musical mediante técnicas de Procesamiento de Lenguaje Natural (NLP). El dataset original contiene aproximadamente cinco millones de registros y abarca seis géneros: country, misc, pop, rap, rb y rock.

Si bien ambos modelos logran identificar patrones lingüísticos relevantes, tienen limitaciones inherentes al uso exclusivo de texto. Las letras por sí solas no capturan todos los elementos musicales (timbre, ritmo, entonación) que definen un género. Para construir un sistema con una mayor presición, sería necesario incorporar el audio de las canciones y aplicar técnicas de visión o procesamiento de señales, como espectrogramas analizados con redes neuronales convolucionales (CNN).

Por restricciones computacionales, los modelos se entrenaron usando solo una fracción del dataset completo:

- Modelo 1: 500,000 registros en Google Colab.
- Modelo 2: 2,200,000 registros en Kaggle.

Estos subconjuntos permitieron entrenar modelos funcionales manteniendo un balance entre rendimiento, tiempos de ejecución y disponibilidad de recursos.

## Limitaciones Generales

Ambos modelos presentan limitaciones importantes derivadas principalmente del tamaño del dataset y de la naturaleza de los datos disponibles. La primera limitación es la falta de recursos computacionales suficientes para procesar los 5,000,000 de registros originales. Debido a ello, fue necesario reducir el tamaño del conjunto de datos a subconjuntos más manejables, lo cual disminuye la diversidad y representatividad de los ejemplos utilizados durante el entrenamiento.

Otra limitación significativa es que el dataset contiene únicamente las letras de las canciones y no incluye información relacionada con el audio. Esto restringe la capacidad del modelo para capturar características musicales complejas como ritmo, tempo, entonación o patrones sonoros propios de cada género. Para mejorar la robustez del sistema y lograr una clasificación más completa, sería necesario integrar tanto el análisis textual como el análisis del audio original, idealmente mediante modelos especializados como redes neuronales convolucionales (CNN) o arquitecturas híbridas.

Finalmente, la falta de metadatos adicionales como estructura musical, producción, instrumentos o información contextual del artista limita el alcance del modelo, que solo puede aprender patrones lingüísticos y no características propias del estilo musical.

## Sesgos del Dataset

El modelo presenta sesgos derivados de la composición del dataset. En primer lugar, el conjunto de datos solo contempla seis géneros musicales, lo cual reduce la diversidad real de los generos musicales. Esta limitación puede generar un sesgo significativo, ya que muchos géneros, subgéneros y fusiones musicales no están representados. Como consecuencia, el modelo aprende a distinguir únicamente entre un conjunto reducido de estilos, lo que limita su capacidad para generalizar si se le proporcionan letras pertenecientes a géneros no incluidos.

No obstante, la reducción a seis categorías también tiene un efecto positivo, ya que disminuye la superposición entre clases y facilita el aprendizaje de patrones específicos, reduciendo la confusión y mejorando la estabilidad del entrenamiento. Aun así, es importante reconocer que esta simplificación puede introducir un sesgo, porque el modelo podría clasificar cualquier letra en uno de los géneros disponibles, incluso si no corresponde a ninguno.

Además, la distribución de los datos entre los géneros puede no ser equilibrada, lo que podría provocar que el modelo favorezca los géneros con mayor cantidad de ejemplos. Esto representa un riesgo de overfitting hacia clases dominantes y una disminución del rendimiento en géneros minoritarios.

## Primer Modelo

El primer modelo implementa una red neuronal recurrente LSTM (Long Short-Term Memory), diseñada para aprender patrones secuenciales presentes en las letras de las canciones.

El preprocesamiento incluye la limpieza del texto mediante expresiones regulares (regex), la tokenización de palabras con un vocabulario de 15,000 términos, la conversión a secuencias numéricas y la aplicación de padding para estandarizar la longitud a 300 tokens.

Se emplea una capa de embedding de 128 dimensiones que aprende representaciones vectoriales de las palabras durante el entrenamiento. La arquitectura LSTM, también con 128 unidades y dropout, permite capturar dependencias temporales en las letras, identificando características lingüísticas y temáticas distintivas de cada género musical.

## Segundo Modelo

El segundo modelo utiliza GloVe (Global Vectors for Word Representation), un algoritmo no supervisado que genera representaciones vectoriales de palabras a partir de grandes corpus de texto. Estas representaciones capturan relaciones semánticas y sintácticas entre las palabras, revelando estructuras lineales en el espacio vectorial.

El modelo combina una CNN y una GRU bidireccional para la clasificación de letras de canciones. Usa los vectores preentrenados de GloVe como embeddings iniciales, lo que permite que el modelo aprenda con mayor rapidez el significado y contexto de las palabras.

La capa convolucional extrae patrones locales y combinaciones de palabras relevantes, mientras que la GRU bidireccional captura dependencias contextuales en ambas direcciones de la secuencia.

El preprocesamiento incluye la limpieza de las letras mediante expresiones regulares (regex), la tokenización de 20,000 palabras, la conversión a secuencias numéricas y la aplicación de padding para limitar las letras a un máximo de 300 tokens.

# Extracción de Datos

El dataset utilizado contiene información recopilada durante 2022 desde la plataforma Genius, un sitio donde los usuarios pueden subir y transcribir letras de canciones, poemas e incluso fragmentos de libros aunque su uso principal está enfocado en la música.

Este conjunto de datos incluye aproximadamente 5 millones de canciones, junto con sus respectivas letras.
Las letras provienen directamente del formato nativo de Genius, el cual requiere un preprocesamiento cuidadoso antes de ser analizado o utilizado para entrenar modelos de Deep Learning.

En particular:

- Las letras suelen incluir metadatos entre corchetes, como [Verse 1], [Chorus] o [Produced by ...], que no forman parte del contenido lírico y deben eliminarse o tratarse.
- La estructura original del texto mantiene los saltos de línea y secciones tal como aparecen en la transcripción, lo cual puede generar dificultades al leer los datos o al pasarlos a modelos que esperan texto plano.
- Además, otras columnas del dataset (como los campos de features o descripciones adicionales) también requieren limpieza y normalización antes de ser utilizadas en el pipeline de procesamiento.

| **Columna**       | **Descripción**                                                                                                                                                |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **title**         | Título de la pieza. La mayoría son canciones, pero también puede haber libros, poemas u otros tipos de contenido.                                              |
| **tag**           | Género de la pieza. La mayoría de obras no musicales se clasifican como _misc_, aunque hay excepciones. Algunas canciones también pueden aparecer como _misc_. |
| **artist**        | Artista, grupo o autor al que se atribuye la pieza.                                                                                                            |
| **year**          | Año de lanzamiento o publicación.                                                                                                                              |
| **views**         | Número de visitas registradas en la página.                                                                                                                    |
| **features**      | Artistas adicionales que participaron o colaboraron en la pieza.                                                                                               |
| **lyrics**        | Letra completa de la canción o texto asociado a la pieza.                                                                                                      |
| **id**            | Identificador único de Genius.                                                                                                                                 |
| **language_cld3** | Idioma de la letra según el modelo CLD3. Si el resultado no es confiable, aparece como `NaN`.                                                                  |
| **language_ft**   | Idioma de la letra según FastText (langid). Los valores con confianza menor a 0.5 se registran como `NaN`.                                                     |
| **language**      | Idioma consolidado solo cuando **CLD3** y **FastText** coinciden; en caso contrario, es `NaN`.                                                                 |

## Tamaño del Dataset

El dataset descargado ocupa 8650.20 MB (aproximadamente 8.44 GB) en disco. Aunque en Kaggle se reporta un tamaño de 9.07 GB, esta diferencia puede deberse a la compresión de archivos o a variaciones en cómo se calcula el tamaño. Con este volumen de datos, se cuenta con información suficiente para entrenar un modelo de deep learning. En este proyecto, se implementará una red neuronal recurrente basada en GRU (Gated Recurrent Unit) para realizar clasificación de textos.

## Limitación de los Registros

Para limitar el tamaño de los datos y optimizar el uso de recursos computacionales durante el entrenamiento y la limpieza del dataset, se seleccionaron únicamente 500,000 filas para el primero modelo 2,200,000 filas para el segundo modelo mediante el parámetro nrows.

Esta reducción permite:

- Acelerar el proceso de carga, preprocesamiento y entrenamiento.
- Disminuir el uso de memoria RAM de la GPU o del entorno y tiempo de cómputo.
- Hacer viable el entrenamiento en Kaggle o Google Colab en un tiempo considerable.

Sin embargo, esta decisión implica una disminución potencial en el rendimiento del modelo, ya que al disponer de menos ejemplos:

- Se reduce la diversidad y representatividad de los datos.
- El modelo puede aprender menos patrones o generalizar peor frente a datos nuevos.

## Selección de features

El dataset original contiene diversas columnas con información sobre las canciones, entre ellas:

`title, tag, artist, year, views, features, lyrics, id, language_cld3, language_ft y language.`

Se determinó que para los fines del proyecto (entrenar un modelo que clasifique las letras según su género musical) no todas las variables son necesarias.

El análisis de frecuencia de géneros muestra que rap es el género predominante en el dataset, seguido por misc, rock, pop, country y finalmente R&B, que presenta la menor representación.

La mayoría de las canciones disponibles en el dataset están en inglés en ambas segmentaciones (Modelo 1 y Modelo 2)

# Transformación de datos

## Filtrado por idioma

Se seleccionan únicamente las canciones cuyo idioma es inglés (language == 'en'), con el propósito de entrenar el modelo con un conjunto de datos lingüísticamente homogéneo y evitar sesgos debidos a diferencias idiomáticas.

El ranking de los géneros más populares en las canciones se mantiene igual al filtrar el lenguaje.

## Normalizar el tamaño de los datos

Con el fin de evitar el sobreajuste (overfitting) observado en el modelo anterior, se identificará el género musical con menor cantidad de datos y se usará como referencia para balancear el tamaño de las demás clases. De esta manera, se entrena el modelo con un conjunto equilibrado, reduciendo el sesgo hacia géneros con más muestras y mejorando su capacidad de generalización.

## Selección de columnas relevantes

Dado que el obetivo es clasificar las letras (lyrics) según su género musical (tag), se conservan únicamente estas dos columnas en el dataset limpio:

- tag: género musical (etiqueta objetivo).
- lyrics: texto de la canción (característica principal para el modelo).

## Limpieza de las letras

Se limpian las letras de las canciones aplicando regex para su transformación, esto nos ayuda ya que los modelos NLP funcionan mejor con texto limpio y normalizado, lo que reduce el ruido y el tamaño del vocabulario:

1. Elimina texto que se encuentra entre corchetes como [Chorus], [Verse 1]... que son metadatos dentro de la letra.

```
r'\[.*?\]'
```

2. Remplaza los saltos de linea por un espacio simple para convertir todo a una sola línea.

```
r'\n+'
```

3. Elimina todos los carácteres, excepto letras (a-z, A-Z) y espacios. Esto ayuda a remover números, signos de puntuación y carácteres especiales.

```
r'[^a-zA-Z\s]'
```

4. Compacta los espacios consecutivos en uno solo.

```
r'\s+'
```

5. Elimina los espacos en blanco al inicio y al final de cada letra.

```
.strip()
```

6. Convierte todo a minúsculas para la normalización de las palabras y tratar a todas las palabras por igual.

```
.lower()
```

```mermaid

flowchart TD
  A[Inicio<br>Texto original de la letra] --> B["Eliminar metadatos entre corchetes<br><code>r'\\[.*?\\]'</code>"]
  B --> C["Unificar saltos de línea<br><code>r'\\n+'</code>"]
  C --> D["Eliminar caracteres no alfabéticos<br><code>r'[^a-zA-Z\\s]'</code>"]
  D --> E["Normalizar espacios consecutivos<br><code>r'\\s+'</code>"]
  E --> F["Eliminar espacios al inicio y final<br><code>.strip()</code>"]
  F --> G["Convertir a minúsculas<br><code>.lower()</code>"]
  G --> H[Fin<br>Texto limpio y normalizado]


```

## Preparar datos para el modelo

1. Variable texts: Convierte la columna de 'lyrics_clean' a una lista de strings. Cada elemento es la letra completa de la canción.
2. Variable labels: Extrae los géneros musicales (ej: "rap", "pop", "rock") como lista.
3. Función LabelEncoder(): Convierte las etiquetas de texto a números, esto es importante porque las redes neuronales solo procesan números.

```mermaid

flowchart TD
  A[Input: Letra limpia] --> B[Agregar a texts]
  B --> C[Obtener genero en texto]
  C --> D[Agregar a labels]
  D --> E[Aplicar LabelEncoder]
  E --> F[Genero convertido a numero]
  F --> G[Output: Etiqueta numerica]

```

```python
texts = df_trimmed['lyrics_clean'].astype(str).tolist()
labels = df_trimmed['tag'].astype(str).tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
```

# Modelado

## Hiperparámetros

**MAX_WORDS** = 20,000 o 15,000: El vocabulario se limitará a 20,000 o 15,000 más frecuentes de las letras y las palabras menos comunes se van a marcar como desconocidas

¿Por qué se eligieron solo 20,000 palabras? Para obtener un balance entre un vocabulario amplio.

**MAX_LEN** = 350 o 300: Cada letra solo va a contener 350 o 300 palabras, si la letra tiene menos de 350 o 300 palabras, entonces se rellena con ceros hasta 350 0 300.

**EMBED_DIM** = 200: Es la dimensión del embedding que empata con la versión que se va a utilizar de Glove, significa que cada palabra se convierte en un vector de 200 valores reales [0.12, -0.45, 0.98, ...] con 200 posiciones.

```python
MAX_WORDS = 20000 | 15000 # Dependiendo del modelo
MAX_LEN = 350 | 300
EMBED_DIM = 200
```

## Cargar glove embeddings en el Modelo 2

Esta función convierte los vectores preentrenados de GloVe en una matriz de embeddings que el modelo puede usar como pesos iniciales en la capa embedding.
Así, cada palabra del vocabulario se asocia con un vector semántico aprendido previamente (en lugar de entrenarlo desde cero).

1. Carga los vectores de Glove, abre el archivo .txt que se va a utilizar. Cada línea del .txt contiene una palabra seguida de sus valores vectoriales.
2. Se guarda en un diccionario 'embedding_index' la palabra y los pesos asociados.
3. Se inicializa una matriz vacía(inicialmente llena de ceros) de embeddings con el tamaño del vocabulario 'num_words' por la dimensión de los embeddings.
4. Recorre todas las palabras del vocabulario que usó el tokenizer y si la palabra se encuentra en Glove, copia su vector en la posición correspondiente de la embedding_matrix.
5. Si no existen palabras similares en Glove, se deja el vector en ceros.
6. Finalmente se devuelve la matriz 'embedding_matrix'

```python

def load_glove_embeddings(glove_file, word_index, embedding_dim):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    num_words = min(MAX_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, i in word_index.items():
        if i >= MAX_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
```

## Dividir datos

Para evaluar de forma justa el rendimiento del modelo, el conjunto de datos se divide en tres subconjuntos:

### Modelo 1

- **Entrenamiento** (80%): usado para ajustar los parámetros del modelo.
- **Prueba** (20%): se reserva completamente para evaluar el desempeño final en datos nunca vistos.

### Modelo 2

- **Entrenamiento** (70%): usado para ajustar los parámetros del modelo.
- **Validación** (15%): permite ajustar hiperparámetros y prevenir sobreajuste durante el entrenamiento.
- **Prueba** (15%): se reserva completamente para evaluar el desempeño final en datos nunca vistos.

random_state=42: Semilla aleatoria que sirve para reproducibilidad de la ejecución.

# Arquitectura por capa

## Modelo 1

Embedding(...): Capa de entrada

- Entrada: Secuencias de números enteros (índices de palabras)
- Salida: Vectores de 128 dimensiones para cada palabra
- input_dim=15000: Vocabulario de 15,000 palabras
- output_dim=128: Cada palabra se convierte en un vector de 128 números
- input_length=300: Cada secuencia tiene 300 palabras
- Función: Convierte índices a vectores densos. Similar a una "tabla de búsqueda" entrenable

#### LSTM(128, ...): Capa recurrente (el "cerebro" del modelo)

- 128 unidades: Tamaño del estado oculto (memoria interna)
- return_sequences=False: Solo devuelve la salida del ÚLTIMO paso de tiempo (no todas las palabras)
- dropout=0.3: Apaga aleatoriamente 30% de las neuronas durante entrenamiento para evitar overfitting
- recurrent_dropout=0.3: Dropout especial para conexiones recurrentes
- Función: Procesa la secuencia palabra por palabra, manteniendo "memoria" del contexto. Captura patrones como "palabras que suelen aparecer juntas en rock" vs "en pop"

#### Dense(64, activation='relu'): Capa densa completamente conectada

- 64 neuronas: Reduce dimensionalidad y aprende combinaciones de características
- relu: Función de activación (deja pasar valores positivos, anula negativos)
- Función: Aprende representaciones de alto nivel a partir de la salida del LSTM

#### Dropout(0.4): Regularización más agresiva

- Apaga 40% de neuronas aleatoriamente
- Función: Previene que el modelo dependa demasiado de características específicas

#### Dense(len(le.classes\_), activation='softmax'): Capa de salida

- Neuronas: Una por cada género (ej: 10 géneros = 10 neuronas)
- softmax: Convierte salidas en probabilidades que suman 1
- Ejemplo de salida: [0.05, 0.70, 0.10, 0.05, 0.10] → 70% probabilidad de ser género #2
- Función: Clasificación final

```mermaid

flowchart TD
    A[Input: Secuencia de 300 indices] --> B[Embedding Layer: convierte indices en vectores densos]
    B --> C[LSTM 128 units: procesa la secuencia y mantiene memoria del contexto]
    C --> D[Dense 64 ReLU: combina caracteristicas y aprende patrones]
    D --> E[Dropout 0.4: regularizacion para evitar sobreajuste]
    E --> F[Dense Softmax: genera probabilidades para cada genero]
    F --> G[Output: Probabilidades de genero]

```

## Modelo 2

#### Embedding(...): Capa de entrada con vectores GloVe

Traduce los índices de palabras a representaciones semánticas ricas, aprovechando el conocimiento previo de GloVe sobre relaciones entre palabras.

- Entrada: Secuencias de números enteros (índices de palabras).
- Salida: Vectores densos de EMBED_DIM dimensiones para cada palabra.
  Parámetros clave:
- input_dim=MAX_WORDS: tamaño del vocabulario (15,000 palabras).
- weights=[embedding_matrix]: inicializa la capa con embeddings preentrenados GloVe.
- trainable=True: permite ajustar ligeramente los vectores durante el entrenamiento.
- input_length=MAX_LEN: longitud fija de cada secuencia (300 tokens).

#### Conv1D(64, 5, activation='relu'): Capa convolucional 1D

Extrae características locales y estructuras comunes, como frases o expresiones típicas de un género musical.

- 64 filtros: detectan patrones locales de palabras (n-gramas) dentro de las secuencias.
- Kernel size = 5: ventana deslizante que capta combinaciones de hasta 5 palabras consecutivas.
- ReLU: activa solo valores positivos, aportando no linealidad.

#### MaxPooling1D(pool_size=2): Capa de reducción

- Reduce la dimensionalidad al conservar solo las características más relevantes de cada filtro.
- Hace que la red sea más eficiente y resistente al ruido en el texto.

#### Dropout(0.3): Regularización

- 30 % de neuronas desconectadas aleatoriamente.
- Evita overfitting, ayudando a que el modelo generalice mejor.

#### Bidirectional(GRU(64, dropout=0.3, recurrent_dropout=0.3)): Capa recurrente bidireccional

Captura el contexto global de la letra, entendiendo cómo una palabra depende de las anteriores y posteriores.

- 64 unidades GRU: cada una procesa dependencias a largo plazo en la secuencia.
- Bidireccional: lee el texto tanto de izquierda a derecha como de derecha a izquierda.
- Dropout y recurrent_dropout = 0.3: previenen sobreajuste tanto en conexiones estándar como recurrentes.

#### Dense(32, activation='relu'): Capa totalmente conectada

Aprende combinaciones de características de alto nivel que representan el estilo o tema de la letra.

- 32 neuronas: condensan la información aprendida por la GRU.
- ReLU: mantiene la no linealidad.

#### Dropout(0.4): Regularización adicional

- 40 % de neuronas desconectadas.
- Refuerza la capacidad de generalización del modelo antes de la clasificación final.

#### Dense(num_classes, activation='softmax'): Capa de salida

- Realiza la clasificación final según la probabilidad de pertenencia a cada género.
- Neuronas: una por cada clase (género musical).
- Ejemplo de salida: [0.05, 0.70, 0.10, 0.05, 0.10] → 70% probabilidad de ser género #2
- Softmax: transforma los valores en probabilidades que suman 1.

```mermaid

flowchart TD
    A[Input: Secuencia de 300 indices] --> B[Embedding GloVe: convierte indices en vectores densos preentrenados]
    B --> C[Conv1D 64 kernel 5 ReLU: extrae patrones locales de la secuencia]
    C --> D[MaxPooling1D: reduce dimensionalidad conservando caracteristicas importantes]
    D --> E[Dropout 0.3: regularizacion para evitar sobreajuste]
    E --> F[Bidirectional GRU 64 units: captura dependencias hacia adelante y hacia atras]
    F --> G[Dense 32 ReLU: combina caracteristicas y aprende patrones]
    G --> H[Dropout 0.4: regularizacion para evitar sobreajuste]
    H --> I[Dense Softmax: genera probabilidades para cada genero]
    I --> J[Output: Probabilidades de genero]

```

# Resultados

## Modelo 1

En el primer modelo se observa que no logró generalizar adecuadamente. La matriz de confusión muestra que las clasificaciones por género no son correctas, a pesar de que el modelo reporta un test accuracy de 0.76. Este valor resulta engañoso cuando se analiza cómo realmente está realizando las predicciones.

La causa principal de este comportamiento fue una mala división de los datos. La mayoría de las letras pertenecen al género rap, por lo que el modelo terminó sesgado hacia esta clase, sin aprender patrones representativos de los demás géneros. El accuracy parece elevado únicamente porque el conjunto está desbalanceado: al haber más ejemplos de un mismo género, el modelo acierta con mayor frecuencia sin necesariamente aprender.

Además de la mala clasificación, el tiempo de entrenamiento fue excesivamente largo: cada epoch tomó alrededor de una hora. Esto indica que tanto la arquitectura del modelo como el método de embedding requieren optimización para reducir el tiempo de cómputo.

Debido a los largos tiempos de ejecución, se utilizó el autoguardado del mejor modelo para continuar el entrenamiento posteriormente. Esto impidió conservar el historial completo de los epochs y, como resultado, no fue posible generar las gráficas de loss y accuracy del proceso de entrenamiento.

| **Clase**        | **Precision** | **Recall** | **F1-score** | **Support** |
| ---------------- | ------------: | ---------: | -----------: | ----------: |
| country          |          0.50 |       0.48 |         0.49 |       2,844 |
| misc             |          0.88 |       0.88 |         0.88 |      15,472 |
| pop              |          0.42 |       0.19 |         0.26 |       9,315 |
| rap              |          0.91 |       0.94 |         0.92 |      35,821 |
| rb               |          0.36 |       0.03 |         0.05 |       1,899 |
| rock             |          0.54 |       0.75 |         0.63 |      14,147 |
| **accuracy**     |             — |          — |   **0.7691** |      79,498 |
| **macro avg**    |          0.60 |       0.55 |         0.54 |      79,498 |
| **weighted avg** |          0.75 |       0.77 |         0.75 |      79,498 |

El modelo muestra un test accuracy de 0.76, pero este valor es altamente engañoso debido al desbalance extremo del dataset. La clase rap domina el conjunto de entrenamiento y prueba, por lo que el modelo aprende a predecir ese género con mucha frecuencia. Esto explica por qué:

- rap tiene alta precision, recall y f1 (0.92)
- clases minoritarias como rb y pop tienen valores muy bajos, especialmente recall
- el macro F1 = 0.54, indicando rendimiento deficiente cuando todas las clases son tratadas por igual

La aparente buena exactitud no refleja una verdadera capacidad de generalización, el modelo está sesgado hacia la clase mayoritaria y no aprende representaciones significativas para las demás.

![alt text](image-1.png)

## Modelo 2

Aunque el segundo modelo alcanza un accuracy de 0.63, generaliza significativamente mejor que el primero. La razón principal es que este modelo incorpora mejoras que permiten aprender representaciones más patrones y evita el overfitting que afectó al primer modelo. Mientras que el primer modelo se sesgó hacia los géneros mayoritarios por una mala distribución de datos, este segundo modelo logra capturar patrones semánticos y estructurales más profundos. Para lograr esto, se balancearon más los datos para que pudiera capturar patrones sin causar overfitting.

La mejora se debe a varios elementos clave de la arquitectura:

- **Embeddings GloVe preentrenados**: El modelo no parte desde cero. Aprovecha conocimiento previo que ya codifica relaciones semánticas entre palabras, lo que le permite distinguir estilos de redacción, temas recurrentes y expresiones típicas de cada género, incluso con datos limitados.
- **Combinación de Conv1D y GRU bidireccional**:
  La capa convolucional detecta patrones locales o frases típicas, mientras que la GRU bidireccional captura el contexto global y las dependencias largas en la letra. Esta combinación aumenta la sensibilidad del modelo al estilo narrativo y estructura del texto.
- **Uso consistente de Dropout**:
  Las capas de regularización evitan que el modelo memorice únicamente los géneros con más ejemplos, reduciendo el overfitting y aumentando la capacidad de generalización.

Gracias a esto, el modelo no solo predice mejor que el anterior, sino que también es más estable, menos sesgado y más representativo del lenguaje usado en cada género musical.

| **Clase**        | **Precision** | **Recall** | **F1-score** | **Support** |
| ---------------- | ------------: | ---------: | -----------: | ----------: |
| country          |          0.63 |       0.79 |         0.70 |       6,104 |
| misc             |          0.84 |       0.70 |         0.76 |       6,104 |
| pop              |          0.51 |       0.21 |         0.30 |       6,104 |
| rap              |          0.75 |       0.82 |         0.79 |       6,104 |
| rb               |          0.58 |       0.68 |         0.63 |       6,104 |
| rock             |          0.49 |       0.60 |         0.54 |       6,104 |
| **accuracy**     |             — |          — |     **0.63** |      36,624 |
| **macro avg**    |          0.63 |       0.63 |         0.62 |      36,624 |
| **weighted avg** |          0.63 |       0.63 |         0.62 |      36,624 |

Este segundo modelo:

- detecta patrones más reales del lenguaje,
- distribuye mejor el recall entre clases,
- mantiene F1 razonables incluso en categorías difíciles,
- y sí generaliza, aunque el accuracy no sea alto.

En clasificación de texto multiclase, un modelo con F1 macro similar a 0.62 ya es un buen avance y además mucho más confiable que el primero.

### ¿Por qué el modelo sigue siendo útil aunque el accuracy sea de 0.63?

En tareas de clasificación de géneros musicales basadas únicamente en texto, es natural que el desempeño sea menor que en modelos que combinan audio y letra. La letra por sí sola no siempre contiene información suficiente para identificar el género con certeza, ya que muchos géneros comparten vocabulario, temas o estructuras narrativas como pasa con el pop. El audio (ritmo, tempo, instrumentos, timbre) suele ser el componente más determinante para distinguir géneros, y en este caso no está disponible.

Aun así, este modelo es útil porque:

- Captura señales textuales reales del estilo y contenido temático, lo que permite sugerir géneros probables basados únicamente en la escritura.
- Generaliza mejor, por lo que sus predicciones se basan en patrones lingüísticos, no en la proporción de datos.
- **Puede servir como componente complementario** en un sistema más amplio si en el futuro se integra audio.

![alt text](image-2.png)

![1763065930710](images/README/1763065930710.png)

## Comparación general entre ambos modelos

| **Categoría**                    | **Modelo 1: LSTM Secuencial**                                               | **Modelo 2: CNN + BiGRU con GloVe**                                                 |
| -------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Tipo de Arquitectura**         | LSTM tradicional que procesa texto paso a paso.                             | Arquitectura híbrida: CNN (patrones locales) + GRU bidireccional (contexto global). |
| **Embedding**                    | Entrenado desde cero (aleatorio → aprendido).                               | Embeddings preentrenados GloVe (200d), semántica fuerte desde el inicio.            |
| **Capacidad Semántica**          | Limitada, depende muchísimo del entrenamiento.                              | Alta, gracias a GloVe + BiGRU.                                                      |
| **Extracción de Patrones**       | Buena en dependencias largas; mala en detectar patrones locales (n-gramas). | CNN detecta patrones locales; BiGRU detecta dependencias largas y globales.         |
| **Direccionalidad del Contexto** | Unidireccional.                                                             | Bidireccional. Mayor comprensión contextual.                                        |
| **Velocidad de Entrenamiento**   | Más lenta; cada epoch ~1 hora.                                              | Más rápida y eficiente, mejor convergencia.                                         |
| **Cantidad de Datos Usados**     | ~500,000 letras                                                             | ~2,200,000 letras                                                                   |
| **Rendimiento Esperado**         | Moderado, sensible a ruido.                                                 | Superior y más estable.                                                             |
| **Fortalezas**                   | Arquitectura simple; aprende dependencias largas.                           | Combina semántica + patrones locales + contexto global.                             |
| **Limitaciones**                 | No aprende bien frases cortas o estructuras locales; embeddings débiles.    | Requiere más memoria; depende del vocabulario de GloVe.                             |
| **Adecuado para…**               | Clasificación básica con poco texto o variabilidad baja.                    | Tareas complejas donde se necesita semántica y patrones más ricos.                  |

## Comparación Final

El primer modelo (LSTM) dependía de aprender semántica desde cero, lo cual es lento, costoso y frágil. Además, el gran desbalance de clases lo sesgó. El segundo modelo partió de embeddings robustos (GloVe), incorporó patrones locales (CNN) y entendió el contexto global en ambas direcciones (BiGRU), logrando una representación más completa, estable y precisa del texto.

# Pasos para correr el proyecto localmente

### Clonar el repositorio

```bash

git clone https://github.com/Angeltrek/SongTextClassifier.git
cd SongTextClassifier

```

Esto descarga todo el código del repositorio localmente.

### Crear y activar un entorno virtual (opcional pero recomendado)

Por ejemplo, usando venv:

```bash

git clone https://github.com/Angeltrek/SongTextClassifier.git
cd SongTextClassifier

```

### Instalar las dependencias

El proyecto incluye un archivo requirements.txt. Una vez activado el entorno virtual, ejecutar:

```bash

pip install -r requirements.txt

```

Esto instalará todo lo necesario para que la app Flask funcione (Flask, bibliotecas de ML, etc.)

### Configurar variables de entorno (si aplica)

— Revisa si en app.py (o en algún otro módulo) se espera alguna variable de entorno (por ejemplo FLASK_ENV, FLASK_APP, configuración de base de datos, rutas, etc.).
— Si necesitas activar modo desarrollo (“debug”), puedes definir:

```bash

export FLASK_ENV=development    # Linux/macOS
set FLASK_ENV=development       # Windows (cmd)
# o en PowerShell: $Env:FLASK_ENV = "development"

```

Con eso, cada cambio en los archivos hará hot-reload del servidor.

### Iniciar la aplicación Flask

Desde la raíz del proyecto, luego de activar el entorno virtual, puedes arrancar la app:

```bash

flask run

```

si app.py define la aplicación como app

```bash

# Unix / macOS
export FLASK_APP=app.py
flask run

# Windows (cmd)
set FLASK_APP=app.py
flask run

# Windows (PowerShell)
$Env:FLASK_APP = "app.py"
flask run

```

Por defecto, la app quedará corriendo en http://127.0.0.1:5000 (o localhost:5000).

### Abrir en el navegador

Ve a http://localhost:5000 (o el host/puerto que te indique el servidor) para ver la interfaz de la aplicación. Si todo está correcto, deberías ver la página principal del proyecto.

![alt text](image.png)
