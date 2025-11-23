# src/scripts/test_gemini_models.py
import google.generativeai as genai
from src.core.config import settings

def test_gemini_models():
    """Prueba qué modelos de Gemini están disponibles"""
    print("🔍 TESTEANDO MODELOS DE GEMINI DISPONIBLES")
    print("=" * 50)
    
    try:
        # Configurar API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Listar modelos disponibles
        print("📋 Modelos disponibles:")
        models = genai.list_models()
        
        gemini_models = []
        for model in models:
            if 'gemini' in model.name.lower():
                gemini_models.append(model.name)
                print(f"  ✅ {model.name}")
                # Mostrar métodos soportados
                if hasattr(model, 'supported_generation_methods'):
                    print(f"     Métodos: {model.supported_generation_methods}")
        
        print(f"\n🎯 Total modelos Gemini: {len(gemini_models)}")
        
        # Probar generación con cada modelo
        print("\n🧪 Probando generación con modelos...")
        test_prompt = "Hola, responde con 'OK' si funciona."
        
        for model_name in gemini_models[:3]:  # Probar solo los primeros 3
            try:
                print(f"\n🔧 Probando: {model_name.split('/')[-1]}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(test_prompt)
                
                if response.text:
                    print(f"  ✅ FUNCIONA: '{response.text.strip()}'")
                else:
                    print(f"  ❌ FALLÓ: {response.prompt_feedback}")
                    
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                
    except Exception as e:
        print(f"❌ Error general: {e}")

if __name__ == "__main__":
    test_gemini_models()