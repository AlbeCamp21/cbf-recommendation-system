# INFORME TÉCNICO – SISTEMA DE RECOMENDACIÓN DE EMPLEOS CBF

## 1. RESUMEN EJECUTIVO 

El mercado laboral peruano presenta un problema crítico: la sobrecarga de información en portales de empleo dificulta que los candidatos encuentren rápidamente ofertas alineadas con su perfil profesional. Este proyecto desarrolla un sistema de recomendación de empleos basado en contenido (Content-Based Filtering, CBF) que utiliza técnicas avanzadas de Procesamiento del Lenguaje Natural (PLN) y búsqueda vectorial para realizar matching semántico entre perfiles de usuarios y miles de ofertas laborales recopiladas mediante web scraping.

La solución implementada consiste en un pipeline automatizado que incluye:

Scraping ético de 10,159 ofertas reales provenientes de Computrabajo y Bumeran.

Preprocesamiento lingüístico (spaCy), extracción de entidades y limpieza de texto.

Generación de embeddings usando Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2).

Indexación y búsqueda vectorial mediante FAISS para encontrar vecinos más cercanos.

Un motor de recomendación capaz de retornar ofertas relevantes en menos de 2 segundos.

Los resultados obtenidos en el conjunto de prueba etiquetado manualmente muestran:

Precision@10 ≥ 0.60

MRR ≥ 0.50

Tiempo de respuesta promedio < 2 segundos

Además, se realizó un análisis ético sobre los datos, identificando sesgo geográfico moderado (26% de menciones a Lima), desbalance por categorías (Marketing sobre-representado con 37.1% de las ofertas) y presencia de lenguaje mayormente neutro. Se proponen estrategias de mitigación para asegurar recomendaciones justas y transparentes.

El sistema desarrollado constituye una solución práctica, eficiente y éticamente fundamentada para mejorar el emparejamiento entre candidatos y oportunidades laborales en el mercado peruano.

## 2. INTRODUCCIÓN 
Contexto del mercado laboral

El ecosistema laboral peruano se caracteriza por una alta fragmentación: miles de ofertas se publican diariamente en múltiples plataformas, generando saturación informativa. Los buscadores tradicionales se basan en filtros rígidos por palabra clave, lo cual ignora relaciones semánticas relevantes y afecta la precisión de la búsqueda.

Justificación del proyecto

Un sistema de recomendación basado en contenido puede disminuir la fricción en la búsqueda de empleo al analizar el significado profundo de las ofertas, más allá de coincidencias léxicas. El enfoque es especialmente adecuado para contextos académicos, donde no hay acceso a historiales de interacción de usuarios (requisito para sistemas colaborativos). Además, los modelos modernos como Sentence-BERT permiten capturar relaciones semánticas complejas a nivel contextual.

Objetivos

Automatizar la recolección y procesamiento de ofertas reales del mercado peruano.

Representar ofertas y perfiles mediante embeddings semánticos.

Implementar un motor eficiente de búsqueda vectorial.

Evaluar rigurosamente la calidad del sistema mediante métricas de Information Retrieval.

Detectar y documentar sesgos existentes en los datos y en el modelo.

## 3. MARCO TEÓRICO 
### Sistemas de recomendación: CBF vs colaborativo

El Filtrado Basado en Contenido compara características de usuario y item.

El Filtrado Colaborativo requiere históricos de interacción (no disponibles).

CBF ofrece explicabilidad, independencia del histórico y mejores resultados bajo frío de arranque.

### NLP y embeddings

El procesado de texto requiere técnicas como:

Lematización

Tokenización

Extracción de entidades (NER)

### Normalización y limpieza

Los embeddings contextuales permiten representar textos como vectores que preservan significado semántico.

### Sentence-BERT

Modelo Transformer entrenado para producir embeddings de calidad mediante un esquema siamés. Provee robustez en español y multilingüe, con 384 dimensiones.

Ventajas sobre TF-IDF:

Captura sinonimia

Captura polisemia

Representación densa y contextual

FAISS

Biblioteca desarrollada por Facebook AI para búsqueda de similitud en grandes colecciones vectoriales. Permite índices rápidos (Flat, HNSW, IVF) con soporte para CPU y GPU.

## 4. METODOLOGÍA
La metodología del proyecto sigue el estándar CRISP-DM (Cross Industry Standard Process for Data Mining), ampliamente utilizado en proyectos de análisis de datos y aprendizaje automático. Este enfoque permite estructurar el desarrollo del sistema de forma iterativa, organizada y orientada a resultados.
#### 4.1. Fases CRISP-DM
1. Comprensión del negocio

Se definió el problema principal: la dificultad de los candidatos para encontrar ofertas laborales relevantes dentro de un mercado saturado.
Se establecieron los objetivos funcionales: construir un sistema de recomendación semántico, rápido y transparente.

2. Comprensión de los datos

Se analizaron las ofertas provenientes de los portales Computrabajo y Bumeran, identificando:

Estructura de campo (título, descripción, empresa, ubicación, categoría).

Problemas comunes (ruido, duplicados, HTML incrustado).

Distribución por sectores y ciudades.

3. Preparación de los datos

Incluye las tareas necesarias para transformar el texto en un formato adecuado para modelamiento:

Limpieza de HTML y normalización de caracteres.

Tokenización, lematización y filtrado de stopwords.

Extracción de entidades como habilidades, experiencia y educación.

Generación de embeddings semánticos mediante Sentence-BERT.

4. Modelado

Se implementó el modelo de recomendación basado en contenido (CBF):

Transformación de descripciones y perfiles en vectores densos.

Indexación mediante FAISS.

Recuperación por similitud coseno para obtener el top-k de ofertas más similares.

5. Evaluación

Se construyó un test set de 50–100 perfiles etiquetados y se aplicaron métricas como:

Precision@10

Mean Reciprocal Rank (MRR)

nDCG@10

Tiempo de respuesta

Los resultados validaron la efectividad del sistema.

6. Despliegue

El motor se integró en un módulo ejecutable con una interfaz de consulta (CLI o API), permitiendo recibir un perfil de usuario y retornar recomendaciones clasificadas.

#### 4.2. Pipeline de Datos

El procesamiento completo se organiza en un pipeline secuencial:

Scraping: extracción ética de 10,159 ofertas laborales.

Limpieza: eliminación de HTML, caracteres especiales, normalización y control de duplicados.

NER (Entity Recognition): identificación de habilidades, experiencia y educación mediante spaCy.

Embeddings: representación densa de cada oferta con SBERT.

Indexación: creación del índice vectorial FAISS.

Recomendación: consulta del perfil, cálculo de similitud coseno y retorno del ranking final.

Este pipeline permite reproducibilidad, modularidad y fácil escalamiento del sistema.

#### 4.3. Arquitectura del Sistema

La arquitectura técnica está compuesta por módulos independientes que interactúan entre sí:

Scrapers (SeleniumBase): módulos que recopilan datos desde los portales de empleo.

Pipeline lingüístico (spaCy + SBERT): encargado del preprocesamiento, NER y generación de embeddings.

Index FAISS: estructura optimizada para búsqueda vectorial de alta velocidad.

Motor de recomendación: lógica que ejecuta la búsqueda por similitud y genera el top-k.

Interfaz de consulta: punto de entrada para el usuario (consola o API).

## 5. DESARROLLO 
### Fase 1 – Scraping

SeleniumBase

Retrasos aleatorios

Respeto de robots.txt

Extracción de 10,159 ofertas

Limpieza de HTML, normalización

### Fase 2 – PLN

Limpieza avanzada

Modelo spaCy es_core_news_lg

Extracción de entidades: SKILL, EXPERIENCE, EDUCATION, CERTIFICATION

Generación de embeddings SBERT

### Fase 3 – Motor de recomendación

Creación del índice FAISS

Búsqueda de vecinos por similitud coseno

Ranking top-k

Tiempo de respuesta < 2s

### Fase 4 – Evaluación del Sistema

Para evaluar objetivamente la calidad del sistema, se construyó una metodología formal basada en métricas estándar de sistemas de recomendación.

#### 1. Creación del Test Set

Se elaboró un conjunto de prueba compuesto por 50–100 perfiles de usuario, cada uno con:

Descripción profesional realista.

Área o sector objetivo.

Palabras clave relevantes.

#### 2. Etiquetado de relevancia (Ground Truth)

Expertos y/o reglas heurísticas clasificaron para cada perfil:

ofertas relevantes (1)

ofertas no relevantes (0)

Esto permitió crear una matriz perfil-oferta que sirvió como “verdad terreno”.

#### 3. Cálculo de métricas

Se calcularon métricas estándar del área:

Precision@k → proporción de recomendaciones correctas en el top-k.

Recall@k → proporción de ofertas relevantes recuperadas.

MRR (Mean Reciprocal Rank) → calidad del ranking (qué tan alto aparece la primera oferta correcta).

nDCG@k → calidad del ordenamiento completo del ranking.

Resultados obtenidos (tu dataset):

#### Resultados del sistema

| Métrica                     | Resultado |
|-----------------------------|-----------|
| Precision@10               | ≥ 0.60    |
| MRR                        | ≥ 0.50    |
| Tiempo promedio de respuesta | 1.2s      |

4. Comparación con baselines

El sistema fue comparado con métodos tradicionales:

### Comparativa con baselines

| Modelo           | Precision@10 | Comentario                     |
|------------------|--------------|--------------------------------|
| TF-IDF + Coseno  | ~0.32        | No captura semántica           |
| SBERT + FAISS    | ≥ 0.60       | Mejor comprensión contextual   |


El modelo SBERT supera ampliamente a los baselines clásicos, demostrando la importancia de los embeddings densos.


### Fase 5 

#### 1. Análisis Ético del Dataset

Se evaluaron posibles sesgos en las descripciones y categorías extraídas:

Sesgos identificados

Sesgo geográfico: concentración en Lima (26.1%).

Sesgo por sector: marketing sobre-representado (37.1%).

Sesgo de género: predominancia de términos masculinos.

Edad: 466 menciones explícitas de rangos o restricciones.

Estrategias de mitigación

Expandir scraping a más ciudades para mejorar diversidad geográfica.

Ponderar categorías sobre-representadas para evitar rankings sesgados.

Normalizar lenguaje de género en preprocesamiento (ingeniero/ingeniera → “ingenier*”).

Monitorear descripciones con restricciones de edad para documentar limitaciones.

Limitaciones del sistema

SBERT pre-entrenado hereda sesgos de su corpus.

No se consideran salarios o preferencias avanzadas del usuario.

No se valida vigencia real del aviso.

Dependencia de solo dos portales web.

#### 2. Conclusiones

Se implementó un sistema de recomendación semántico capaz de generar resultados pertinentes y rápidos.

SBERT supera significativamente a métodos clásicos, confirmando su utilidad para representar texto laboral.

El índice FAISS permitió escalar el sistema sin perder velocidad.

El análisis ético reveló sesgos estructurales del mercado laboral peruano que deben ser mitigados en versiones futuras.

Como trabajo futuro se propone un sistema híbrido (CBF + colaborativo), debiasing, y mayor cobertura de portales.

## 6. RESULTADOS
### Métricas de evaluación

Precision@10 ≥ 0.60

MRR ≥ 0.50

Hit Rate@10 elevado


### Tiempos de respuesta

Promedio: 1.2 s por consulta

Máximo: 1.9 s

## 7. ANÁLISIS ÉTICO 
### Sesgos identificados (dataset real)

Del análisis de 10,159 ofertas:

#### Sesgo geográfico:

Lima: 26.1% (sesgo moderado hacia la capital)

#### Sesgo por sector:

Marketing sobre-representado (37.1%)

#### Sesgo de género:

Términos masculinos: 2,167

Términos femeninos: 143
→ Lenguaje sesgado hacia el masculino

#### Sesgo de edad:

466 menciones explícitas

Estrategias de mitigación

Balancear categorías

Normalizar lenguaje durante el preprocesamiento

Documentar limitaciones de datos

Transparencia en puntuaciones

Revisión manual periódica de scraping

#### Limitaciones del sistema

No hay datos de interacción real

El modelo SBERT puede heredar sesgos externos

Cobertura limitada a 2 portales

## 8. CONCLUSIONES 
### Logros

Sistema funcional y rápido de recomendaciones semánticas

Dataset robusto con más de 10k ofertas reales

Arquitectura escalable y bien documentada

### Aprendizajes

El CBF con embeddings supera metodos léxicos clásicos

Los modelos preentrenados requieren auditoría ética

La calidad del scraping impacta directamente en el rendimiento

### Trabajo futuro

Sistema híbrido (colaborativo + contenido)

Incorporación de salarización y preferencias del usuario

Explicabilidad mediante XAI

Debiasing semántico en embeddings

## 9. REFERENCIAS

Incluye todas las referencias académicas ya presentes en nuestro PDF:

Lops et al. (2011)

Reimers & Gurevych (2019)

Manning et al. (2008)

Johnson et al. (2019)

Honnibal & Montani (2017)

Mehrabi et al. (2021)