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
from fastapi import FastAPI, HTTPException, File, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from nltk.stem import PorterStemmer

# Se inicializa FastAPI
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Se cargan los stopwords y otros recursos de NLTK
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('spanish')
wpt = nltk.WordPunctTokenizer()
ps = PorterStemmer()

#Se carga el archivo de datos utilizado en la Etapa1
file_path = r"data\ODScat_345.xlsx"  
try:
    data_t = pd.read_excel(file_path, engine='openpyxl')
except:
    data_t = pd.read_excel(file_path, engine='xlrd') 

#Se adapta la función replace_special_chars para los datos utilizados en la Etapa1
def replace_special_chars(df):
    replacements = {
        'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
        'Ã': 'Á', 'Ã': 'É', 'Ã': 'Í', 'Ã': 'Ó', 'Ã': 'Ú',
        'Ã±': 'ñ', 'Ã': 'Ñ', 'Ã¼': 'ü', 'Ã': 'Ü'
    }
    for col in df.select_dtypes(include=['object']).columns:
        for old, new in replacements.items():
            df[col] = df[col].str.replace(old, new)
    return df
#Se envia a un excel los datos que se van a utilizar para el reentrenamiento
data_t.to_excel('data/ODScat_345_1.xlsx', index=False)
# Se aplica la función a los datos cargados
data_t = replace_special_chars(data_t)

# Se devide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data_t['Textos_espanol'], data_t['sdg'], test_size=0.2, random_state=42)

# Se crea el pipeline para automatizar el proceso
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),  # Vectorización con TF-IDF
    ('svm', SVC(kernel='rbf', gamma='scale', C=1, probability=True))  # Clasificador SVM
])

# Se entrena el pipeline
pipeline.fit(X_train, y_train)

# Se guarda el pipeline completo (preprocesamiento + modelo entrenado)
joblib.dump(pipeline, 'models/modelo_analitico.pkl')



# Se adapta la función replace_special_chars para listas de textos
def replace_special_chars(texts):
    replacements = {
        'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
        'Ã': 'Á', 'Ã': 'É', 'Ã': 'Í', 'Ã': 'Ó', 'Ã': 'Ú',
        'Ã±': 'ñ', 'Ã': 'Ñ', 'Ã¼': 'ü', 'Ã': 'Ü', 'ã³':'ó', 'ã­':'í',
        'ã©':'é', 'ã¡':'á', 'ãº':'ú', 'ã°':'ó', 'ã':'ñ', 'ã':'ü'
    }

    # Se reemplazan los caracteres especiales en cada texto
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

# Se normaliza cada texto individualmente
def normalize_corpus(texts):
    return [normalize_documents(text) for text in texts]

# Función para preprocesar los textos
def preprocess_texts(texts):
    # Se reemplazan los caracteres especiales
    textos_reemplazados = replace_special_chars(texts)
    # Se aplica normalización y demás procesamiento a los datos
    textos_normalizados = normalize_corpus(textos_reemplazados)
    return textos_normalizados

# Se carga el modelo entrenado
modelo_entrenado = joblib.load('models/modelo_analitico.pkl')

# Endpoint de predicción con cargue de archivo CSV
@app.post("/prediccion/")
async def predecir(file: UploadFile = File(...)):
    try:
        # Car el archivo CSV
        df = pd.read_csv(file.file, sep=',', encoding = "ISO-8859-1")

        textos = df['Textos_espanol'].tolist()

        # Se preprocesan los textos antes de predecir
        textos_preprocesados = preprocess_texts(textos)

        # Se realiza la predicción con el modelo
        predicciones = modelo_entrenado.predict(textos_preprocesados)
        probabilidades = modelo_entrenado.predict_proba(textos_preprocesados)
        
        resultado = []
        for i in range(len(predicciones)):
            resultado.append({
                "prediccion": int(predicciones[i]),  
                "probabilidad": float(max(probabilidades[i]))*100
            })

        return {"resultado": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint de reentrenamiento con cargue de archivo CSV
@app.post("/reentrenamiento/")
async def reentrenar(file: UploadFile = File(...)):
    try:
        #Se carga el archivo de datos utilizado en el último reentrenamiento
        file_path_p = r"data\ODScat_345_1.xlsx"  
        try:
            data_p = pd.read_excel(file_path_p, engine='openpyxl')
        except:
            data_p = pd.read_excel(file_path_p, engine='xlrd') 

        # Se carga el archivo CSV
        df1 = pd.read_csv(file.file, sep=',', encoding = "ISO-8859-1")
        #Se concatena el archivo cargado con datos nuevos con el archivo de datos utilizado en la Etapa1 para el reentrenamiento
        df= pd.concat([data_p, df1], ignore_index=True)
        df.to_excel('data/ODScat_345_1.xlsx', index=False)
        textos = df['Textos_espanol'].tolist()
        etiquetas = df['sdg'].tolist()

        # Se preprocesan los textos antes de reentrenar
        textos_preprocesados = preprocess_texts(textos)

        # Se reentrena el modelo
        modelo_entrenado.fit(textos_preprocesados, etiquetas)
        joblib.dump(modelo_entrenado, 'models/modelo_analitico.pkl')

        # Se realizan las predicciones para calcular las métricas
        y_pred = modelo_entrenado.predict(textos_preprocesados)
        exactitud = accuracy_score(etiquetas, y_pred)
        precision = precision_score(etiquetas, y_pred, average='weighted')
        recall = recall_score(etiquetas, y_pred, average='weighted')
        f1 = f1_score(etiquetas, y_pred, average='weighted')

        return {
            "exactitud": exactitud,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Se ejecuta la api
if __name__ =="__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

#COMANDOS PARA EJECUTAR EL PROYECTO

#Comandos para ejecutar el entorno virtual
#python -m venv env
#env\Scripts\activate

# Comandos para instalar las librerías necesarias
#pip install uvicorn
#pip install nltk
#pip install fastapi[all]
#pip install python-multipart
#pip install pandas
#pip install joblib
#pip install scikit-learn
#pip install openpyxl
#pip install numpy

# Comandos para ejecutar la API
#uvicorn api:app --reload
