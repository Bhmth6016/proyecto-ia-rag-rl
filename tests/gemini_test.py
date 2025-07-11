from vertexai.language_models import ChatModel
from vertexai.preview.language_models import InputOutputTextPair
import vertexai
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/x/OneDrive/Documents/GitHub/Nueva carpeta/proyecto-ia-rag-rl/proyecto-ia-rag-rhlf-465606-5ca37ec532b6.json"


# Inicializa Vertex AI con tu proyecto y ubicación
vertexai.init(
    project="proyecto-ia-rag-rhlf",
    location="us-central1"
)


# Cargar el modelo Gemini Pro
chat_model = ChatModel.from_pretrained("chat-bison@001")  # puedes usar también "gemini-pro"

# Crear una sesión de chat
chat = chat_model.start_chat()

# Enviar un mensaje
response = chat.send_message("¿Cuál es la capital de Francia?")
print("Gemini responde:", response.text)
