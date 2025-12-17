# scripts/check_ollama.py - Verifica y ayuda a configurar Ollama

import requests
import subprocess
import sys
import os
from pathlib import Path

def check_ollama():
    """Verifica el estado de Ollama y ayuda a configurarlo."""
    
    print("🔍 Verificando configuración de Ollama...")
    print("=" * 60)
    
    # Verificar si Ollama está instalado
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama instalado: {result.stdout.strip()}")
        else:
            print("❌ Ollama no parece estar instalado o no está en PATH")
            print("\n💡 Para instalar Ollama:")
            print("   1. Visita https://ollama.ai/")
            print("   2. Descarga e instala Ollama")
            print("   3. Asegúrate de que 'ollama' esté en tu PATH")
            return False
    except FileNotFoundError:
        print("❌ Ollama no encontrado. Por favor instálalo desde https://ollama.ai/")
        return False
    
    # Verificar si el servicio Ollama está corriendo
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Servicio Ollama está corriendo en http://localhost:11434")
        else:
            print(f"⚠️  Ollama responde con código {response.status_code}")
            print("💡 Intenta: ollama serve (en otra terminal)")
            return False
    except requests.ConnectionError:
        print("❌ No se puede conectar a Ollama en http://localhost:11434")
        print("\n💡 Soluciones:")
        print("   1. Asegúrate de que Ollama esté corriendo: ollama serve")
        print("   2. Verifica que el puerto 11434 no esté bloqueado")
        return False
    
    # Verificar modelos disponibles
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                print(f"✅ Modelos disponibles ({len(models)}):")
                for model in models:
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0) / (1024**3)  # Convertir a GB
                    print(f"   • {name} ({size:.1f}GB)")
            else:
                print("⚠️  No hay modelos descargados")
                print("\n💡 Para descargar un modelo:")
                print("   ollama pull llama3.2:3b  # Modelo pequeño y rápido")
                print("   ollama pull llama3.2:1b  # Modelo muy pequeño")
    except Exception as e:
        print(f"⚠️  Error verificando modelos: {e}")
    
    print("\n🎯 Configuración recomendada para este proyecto:")
    print("   • Modelo: llama-3.2-3b-instruct (equilibrado)")
    print("   • Endpoint: http://localhost:11434")
    print("   • Temperature: 0.1 (respuestas más deterministas)")
    
    return True

def setup_ollama_model(model_name="llama-3.2-3b-instruct"):
    """Intenta descargar el modelo si no está disponible."""
    
    print(f"\n📥 Configurando modelo: {model_name}")
    
    # Verificar si ya está descargado
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name') for m in models]
            
            if model_name in model_names:
                print(f"✅ Modelo {model_name} ya está descargado")
                return True
    except:
        pass
    
    # Preguntar si descargar
    print(f"⚠️  El modelo {model_name} no está descargado")
    response = input(f"¿Descargar {model_name}? (s/n): ").strip().lower()
    
    if response == 's':
        try:
            print("⏳ Descargando modelo... Esto puede tomar unos minutos.")
            print("   (Depende de tu conexión a internet)")
            
            result = subprocess.run(['ollama', 'pull', model_name], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Modelo {model_name} descargado exitosamente")
                return True
            else:
                print(f"❌ Error descargando modelo: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    else:
        print("⚠️  Saltando descarga del modelo")
        return False

if __name__ == "__main__":
    print("🦙 Configurador de Ollama para RAG E-commerce")
    print("=" * 60)
    
    # Verificar Ollama
    if check_ollama():
        # Configurar modelo
        setup_ollama_model("llama-3.2-3b-instruct")
        
        print("\n🎉 Configuración completada!")
        print("\n💡 Ahora puedes ejecutar:")
        print("   python main.py rag --mode enhanced --ml")
        print("\n⚠️  Si prefieres no usar LLM, ejecuta sin --ml")
        print("   python main.py rag --mode basic")
    else:
        print("\n❌ Ollama no está configurado correctamente.")
        print("💡 Para usar el sistema sin LLM:")
        print("   python main.py rag --mode basic")
        print("   python main.py rag --mode enhanced --no-ml")