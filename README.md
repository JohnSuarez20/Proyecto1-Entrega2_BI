# Proyecto1-Entrega2_BI

Como desplegar la aplicacion:

1. Ubicados en la carpeta raíz del proyecto y con la ayuda de una consola ejecutamos el entorno virtual:

# Comandos para ejecutar el entorno virtual
python -m venv env
env\Scripts\activate

2. Nos aseguramos que tengamos todas las dependencias y librerias instaladas:

# Comandos para instalar las librerías necesarias
pip install uvicorn
pip install nltk
pip install fastapi[all]
pip install python-multipart
pip install pandas
pip install joblib
pip install scikit-learn
pip install openpyxl
pip install numpy

3. Se ejecuta la API:

# Comandos para ejecutar la API
uvicorn api:app --reload

4. Por ultimo abrimos el html en el que se encuentra en la carpeta frontend de la aplicación. Basta con hacer doble click en el archivo 'index.html'.

