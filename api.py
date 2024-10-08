import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Cargar el archivo de datos
file_path = r"data\ODScat_345.xlsx"
data_t = pd.read_excel(file_path, engine='openpyxl')

# Reemplazar caracteres especiales
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

data_t = replace_special_chars(data_t)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(data_t['Textos_espanol'], data_t['sdg'], test_size=0.2, random_state=42)

# Crear el pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('svm', SVC(kernel='rbf', gamma='scale', C=1))
])

# Entrenar el pipeline
pipeline.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(pipeline, 'models/modelo_analitico.pkl')

# Cargar el modelo entrenado
modelo_entrenado = joblib.load('models/modelo_analitico.pkl')

# API con FastAPI
app = FastAPI()

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
        predicciones = modelo_entrenado.predict(datos.textos)
        return {"predicciones": predicciones.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint de reentrenamiento
@app.post("/reentrenamiento/")
def reentrenar(datos: DatosReentrenamiento):
    try:
        modelo_entrenado.fit(datos.textos, datos.etiquetas)
        joblib.dump(modelo_entrenado, 'models/modelo_analitico.pkl')
        return {"mensaje": "Modelo reentrenado con éxito"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecucióm la API con Uvicorn
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

#uvicorn api:app --reload