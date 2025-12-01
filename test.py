# test_gemini.py - Crear archivo de prueba
from src.core.init import get_system

def test_gemini_integration():
    system = get_system()
    llm = system.llm_model
    
    if llm:
        print("✅ Gemini configurado correctamente")
        # Probar generación
        response = llm.generate_content("Recomienda un juego de acción")
        print(f"Respuesta: {response.text}")
    else:
        print("❌ Gemini no está configurado")
        print("💡 Verifica:")
        print("  1. Variable GEMINI_API_KEY en .env")
        print("  2. pip install google-generativeai")
        print("  3. La API key es válida")

if __name__ == "__main__":
    test_gemini_integration()