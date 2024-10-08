import pandas as pd
import joblib
import re
import nltk
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from nltk.stem import PorterStemmer

# Inicializar FastAPI
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar los stopwords y otros recursos de NLTK
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('spanish')
wpt = nltk.WordPunctTokenizer()
ps = PorterStemmer()

# Adaptar la función replace_special_chars para listas de textos
def replace_special_chars(texts):
    # Diccionario de reemplazo
    replacements = {
        'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
        'Ã': 'Á', 'Ã': 'É', 'Ã': 'Í', 'Ã': 'Ó', 'Ã': 'Ú',
        'Ã±': 'ñ', 'Ã': 'Ñ', 'Ã¼': 'ü', 'Ã': 'Ü', 'ã³':'ó', 'ã­':'í',
        'ã©':'é', 'ã¡':'á', 'ãº':'ú', 'ã°':'ó', 'ã':'ñ', 'ã':'ü'
    }

    # Reemplazar caracteres especiales en cada texto
    texts_replaced = []
    for text in texts:
        for old, new in replacements.items():
            text = text.replace(old, new)
        texts_replaced.append(text)

    return texts_replaced

# Función para normalizar documentos
def normalize_documents(doc):
    """Normaliza un documento preservando acentos y capitalización adecuada"""
    doc = re.sub(r'[^\w\sáéíóúÁÉÍÓÚñÑ]', '', doc)  # Mantiene tildes y caracteres como ñ
    words = doc.split()
    new_words = []
    capitalize_next = True

    for word in words:
        if capitalize_next:
            new_word = word
            capitalize_next = False
        else:
            if word[0].isupper() and word[1:].islower():
                new_word = word
            else:
                new_word = word.lower()

        if word.endswith('.'):
            capitalize_next = True

        new_words.append(new_word)

    doc = ' '.join(new_words)
    tokens = wpt.tokenize(doc)
    filtered_token = [ps.stem(token) for token in tokens if token.lower() not in stop_words]
    doc = ' '.join(filtered_token)
    return doc

# Normalizar un conjunto de documentos
def normalize_corpus(texts):
    return [normalize_documents(text) for text in texts]

# Función para preprocesar los textos
def preprocess_texts(texts):
    # Primero reemplazar caracteres especiales
    textos_reemplazados = replace_special_chars(texts)
    # Luego aplicar normalización y demás procesamiento
    textos_normalizados = normalize_corpus(textos_reemplazados)
    return textos_normalizados

# Cargar el modelo entrenado
modelo_entrenado = joblib.load('models/modelo_analitico.pkl')

# Esquemas de entrada
class DatosPrediccion(BaseModel):
    textos: list

class DatosReentrenamiento(BaseModel):
    textos: list
    etiquetas: list

# Endpoint de predicción
@app.post("/prediccion/")
def predecir(datos: DatosPrediccion):
    try:
        # Preprocesar los textos antes de predecir
        textos_preprocesados = preprocess_texts(datos.textos)

        # Realizar la predicción con el modelo
        predicciones = modelo_entrenado.predict(textos_preprocesados)
        return {"predicciones": predicciones.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint de reentrenamiento
@app.post("/reentrenamiento/")
def reentrenar(datos: DatosReentrenamiento):
    try:
        # Preprocesar los textos antes de reentrenar
        textos_preprocesados = preprocess_texts(datos.textos)

        # Reentrenar el modelo
        modelo_entrenado.fit(textos_preprocesados, datos.etiquetas)
        joblib.dump(modelo_entrenado, 'models/modelo_analitico.pkl')

        # Realizar predicciones para calcular las métricas
        y_pred = modelo_entrenado.predict(textos_preprocesados)
        precision = precision_score(datos.etiquetas, y_pred, average='weighted')
        recall = recall_score(datos.etiquetas, y_pred, average='weighted')
        f1 = f1_score(datos.etiquetas, y_pred, average='weighted')

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecutar la API con Uvicorn
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

#pip install nltk
#pip install fastapi[all]
#uvicorn api:app --reload