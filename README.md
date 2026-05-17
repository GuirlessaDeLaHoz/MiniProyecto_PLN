````markdown
# Resume Classification using NLP and DistilBERT

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)

---

# Descripción del proyecto

Este proyecto implementa un sistema de clasificación automática de currículums vitae utilizando técnicas de Procesamiento de Lenguaje Natural (NLP), Machine Learning y Deep Learning.

El objetivo principal es predecir automáticamente la categoría profesional de un currículum a partir de su contenido textual.

El proyecto incluye:

- Análisis exploratorio de datos (EDA)
- Preprocesamiento NLP
- Modelos clásicos de Machine Learning
- Modelos Deep Learning
- Transformers (DistilBERT)
- API REST con FastAPI
- Contenerización con Docker
- Testing automatizado con Pytest

---

# Objetivos

- Realizar análisis exploratorio de datos (EDA)
- Implementar pipelines de preprocesamiento NLP
- Comparar modelos clásicos y Deep Learning
- Desplegar el mejor modelo mediante FastAPI
- Contenerizar la aplicación con Docker
- Implementar pruebas básicas para validación

---

# Dataset

El dataset contiene currículums clasificados en múltiples categorías profesionales como:

- Data Science
- HR
- Advocate
- Mechanical Engineer
- Web Designing
- Java Developer
- Information Technology
- Business Analyst
- Civil Engineer
- Sales
- etc.

---

# Pipeline NLP

## Preprocesamiento aplicado

- Conversión a minúsculas (Lowercasing)
- Eliminación de puntuación
- Eliminación de stopwords
- Tokenización
- Padding de secuencias
- Embeddings semánticos

---

# Modelos implementados

| Modelo | Tipo |
|---|---|
| TF-IDF + XGBoost | Machine Learning |
| CNN-1D | Deep Learning |
| FastText + BiLSTM | Embeddings + RNN |
| Word2Vec + BiLSTM | Embeddings + RNN |
| DistilBERT | Transformer |

---

# Resultados comparativos

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + XGBoost | 0.4933 | 0.5781 | 0.4933 | 0.4441 | 0.8943 |
| CNN-1D | 0.5131 | 0.4788 | 0.5131 | 0.4772 | 0.8933 |
| FastText + BiLSTM | 0.6761 | 0.6577 | 0.6761 | 0.6569 | 0.9385 |
| Word2Vec + BiLSTM | 0.6901 | 0.6691 | 0.6901 | 0.6732 | 0.9351 |
| DistilBERT | 0.8231 | 0.8251 | 0.8231 | 0.8195 | 0.9695 |

---

# Interpretación de resultados

- DistilBERT obtuvo el mejor desempeño general.
- Los modelos Transformer superaron a los enfoques tradicionales basados en TF-IDF.
- Word2Vec y FastText mostraron mejor capacidad semántica que CNN-1D.
- TF-IDF + XGBoost presentó limitaciones para capturar contexto semántico complejo.

---

# Trade-offs: precisión vs costo computacional

| Modelo | Precisión | Tiempo entrenamiento | Consumo recursos |
|---|---|---|---|
| TF-IDF + XGBoost | Bajo | Bajo | Bajo |
| CNN-1D | Medio | Medio | Medio |
| FastText + BiLSTM | Alto | Alto | Medio |
| Word2Vec + BiLSTM | Alto | Alto | Medio |
| DistilBERT | Muy alto | Muy alto | Alto |

---

# API REST

## Endpoint

POST /predict_text

## Request

```json
{
  "text": "Experienced data scientist with Python and machine learning skills"
}
````

## Response

```json
{
  "prediction": "INFORMATION-TECHNOLOGY",
  "confidence": 0.1651,
  "inference_time_seconds": 0.0712
}
```

---

# Instalación

## Clonar repositorio

```bash
git clone https://github.com/tuusuario/MiniProject_PLN.git
```

## Crear entorno virtual

```bash
python -m venv venv
```

## Activar entorno virtual

### Linux / WSL

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Ejecutar API localmente

```bash
uvicorn app.main:app --reload
```

---

# Docker

## Construir imagen Docker

```bash
docker build -t resume-api .
```

## Ejecutar contenedor Docker

```bash
docker run -p 8000:8000 resume-api
```

## Swagger UI

[http://localhost:8000/docs](http://localhost:8000/docs)

---

# Tests

## Ejecutar pruebas automatizadas

```bash
python -m pytest
```

---

# Monitoreo en producción

* Latencia de inferencia
* Accuracy en producción
* Data Drift
* Distribución de clases

---

# Estructura del proyecto

```text
MiniProject_PLN/
├── data/
├── notebooks/
├── app/
├── models/
├── test/
├── Dockerfile
├── requirements.txt
├── requirements_api.txt
└── README.md
```

---

# Limitaciones

* Dataset relativamente pequeño para Transformers
* Desbalance entre clases
* Inferencia más costosa en DistilBERT

---

# Conclusiones

## Modelo recomendado para producción

DistilBERT fue seleccionado debido a su superior desempeño en métricas de clasificación y capacidad de generalización.

## Mejoras futuras

* Fine-tuning más profundo
* Balanceo de clases
* Optimización de inferencia
* Cuantización del modelo

---

# Tecnologías utilizadas

* Python
* PyTorch
* Transformers
* Scikit-Learn
* FastAPI
* Docker
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Pytest

```
```
