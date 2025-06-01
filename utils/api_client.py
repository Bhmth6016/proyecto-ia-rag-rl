from googleapiclient.discovery import build
from google.oauth2 import service_account

# Ruta al archivo de credenciales JSON descargado
SERVICE_ACCOUNT_FILE = "ruta/a/tu/archivo-credenciales.json"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# ID del proyecto y ubicación (por ejemplo, "us-central1")
PROJECT_ID = "tu-proyecto"
LOCATION = "us-central1"

def generar_recomendaciones_gemini(prompt):
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build("cloudaicompanion", "v1", credentials=credentials)

    name = f"projects/{PROJECT_ID}/locations/{LOCATION}"

    # Aquí iría la llamada al método correcto, por ejemplo:
    # (Reemplaza esto por el método específico de generación de texto)
    request = service.projects().locations().someMethod(
        name=name,
        body={
            "prompt": prompt,
            "maxTokens": 100,
            # Otros parámetros...
        }
    )
    response = request.execute()

    # Procesar la respuesta para extraer recomendaciones
    recomendaciones = response.get("recommendations", [])
    return recomendaciones


if __name__ == "__main__":
    test_prompt = "Productos recomendados para entrenamiento en casa"
    resultados = generar_recomendaciones_gemini(test_prompt)
    print(resultados)
