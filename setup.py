#!/usr/bin/env python3
# setup.py - Configuración inicial del proyecto

import os
from pathlib import Path

def create_directory_structure():
    """Crea la estructura de directorios necesaria"""
    dirs = [
        'data/raw',
        'data/processed',
        'logs',
        'configs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Directorio creado: {dir_path}")

    # Crear archivos de configuración básicos si no existen
    if not Path('.env').exists():
        with open('.env', 'w') as f:
            f.write("# Configuración del entorno\n")
            f.write("OPENAI_API_KEY=tu_clave_aqui\n\n")
        print("Archivo .env creado - Por favor completa tus credenciales")

if __name__ == "__main__":
    print("Configurando proyecto Amazon Recommendation System...")
    create_directory_structure()
    print("\nConfiguración completada. Por favor:")
    print("1. Completa el archivo .env con tus credenciales")
    print("2. Coloca tus datos en data/raw/")
    print("3. Ejecuta 'python main.py' para iniciar el sistema")