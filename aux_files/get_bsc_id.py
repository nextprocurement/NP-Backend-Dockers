import pandas as pd
import requests

df_PLACE_all_processed_augmented_for_np_backend = pd.read_parquet("/export/usuarios_ml4ds/lbartolome/NextProcurement/sproc/place_feb_21/place_all_embeddings_for_np_backend.parquet")

def obtener_id_place(procurement_id_x):
    url_base = "https://nextprocurement.bsc.es/api/place/"
    url_fallback = "https://nextprocurement.bsc.es/api/place_menores/"
    url_completa = f"{url_base}{procurement_id_x}/id"
    url_completa_fallback = f"{url_fallback}{procurement_id_x}/id"
    
    try:
        response = requests.get(url_completa)
        print(url_completa)        
        if response.status_code == 200:
            data = response.json()
            place_id = data.get('id')
            return place_id
    except Exception as e:
        print("Excepción durante la solicitud con la URL principal:", str(e))

    try:
        response = requests.get(url_completa_fallback)
        
        if response.status_code == 200:
            data = response.json()
            print("El json completo es:", data)  
            place_id = data.get('id')
            print("ID obtenido:", place_id)  
            return place_id
    except Exception as e:
        print("Excepción durante la solicitud con la URL secundaria:", str(e))
    return None