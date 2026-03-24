#aquí es para traer los cambios hechos en Atlas con Colab (poner embeddings de Bert) y seguir trabajando en el Local.
#se ejecuta solo después de haber migrado los datos en la nube corrido el colab en la nube.
#solo se agregó el cls desde el atlas, no hay ningún otro cambio.
from dotenv import load_dotenv #para el archivo .env
from pymongo import MongoClient

load_dotenv() #para el Mongo Atlas
# 1. CONEXIONES
client_atlas = MongoClient("MONGO_URI")
client_local = MongoClient("mongodb://localhost:27017/")

# 2. DEFINIR DB (Usa el nombre que ya tenías para no cambiar rutas luego)
db_atlas = client_atlas['dbx_Canciones']
db_local = client_local['dbx_Canciones']  # Mismo nombre que en PyCharm

col_atlas = db_atlas['canciones']
col_local = db_local['canciones']

print("Iniciando migración de 10k canciones...")

try:
    documentos = list(col_atlas.find())

    if documentos:
        # Insertamos en local
        col_local.insert_many(documentos)
        print(f"Listo {len(documentos)}")
    else:
        print("No hay datos en Atlas.")
except Exception as e:
    print(f"Error: {e}")