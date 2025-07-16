Creación del Ambiente para el Proyecto de Recomendación de Amazon
A continuación te detallo cómo crear un ambiente completo para tu proyecto, incluyendo la configuración del entorno virtual, instalación de dependencias, estructura de directorios y variables de entorno.

1. Estructura de Directorios Completa
amazon-recommendation-system/
├── .env                    # Archivo para variables de entorno
├── .gitignore
├── README.md
├── requirements.txt        # Dependencias principales
├── requirements_dev.txt    # Dependencias para desarrollo
├── main.py                 # Punto de entrada principal
├── data/
│   ├── raw/                # Datos brutos en JSON/JSONL
│   └── processed/          # Datos procesados e índices
├── configs/
│   └── rlhf_config.yaml    # Configuración para RLHF
├── demo/
│   └── generator.py        # Script para generación de datos
├── src/                    # (Todo tu código fuente existente)
└── tests/                  # Tests unitarios

2. Configuración del Entorno Virtual
Ejecuta los siguientes comandos en tu terminal:
# Crear entorno virtual (Python 3.8+ recomendado)
python -m venv venv

# Activar el entorno (Windows)
venv\Scripts\activate

# Actualizar pip
pip install --upgrade pip

3. Verificar que se tenga el archivo requeriments.txt
# Core dependencies
langchain==0.1.13
openai==1.12.0
python-dotenv==1.0.0
faiss-cpu==1.7.4
chromadb==0.4.22
numpy==1.26.4

# Procesamiento de datos
pydantic==2.6.4
pandas==2.2.1
tqdm==4.66.2

# Interfaz
textual==0.54.0
rich==13.7.1

# Machine Learning
torch==2.7.1
transformers==4.38.2
sentence-transformers==2.5.1
peft==0.9.0
trl==0.7.11

4. Verificar que se tenga el archivo requirements_dev.txt
En la carpeta raíz del proyecto, actualizado sino.
crea un archivo llamado requirements.txt y agrega las siguientes líneas:
# Testing
pytest==8.0.2
pytest-cov==4.1.0
pytest-mock==3.12.0

# Linting and formatting
black==24.2.0
flake8==7.0.0
mypy==1.8.0
isort==5.13.2

# Documentation
mkdocs==1.5.3
mkdocs-material==9.5.3

# Jupyter (para experimentación)
jupyter==1.0.0
ipython==8.22.2

5. Archivo .env
Crea este archivo en la raíz con tus claves API:
# OpenAI
OPENAI_API_KEY=tu_clave_aqui

# Configuración de ChromaDB
CHROMA_DB_PATH=./data/processed/chroma_db
CHROMA_DB_COLLECTION=amazon_products

# Configuración de logging
LOG_LEVEL=INFO
LOG_FILE=./logs/amazon_recommendations.log

# Límites del sistema
MAX_PRODUCTS_TO_LOAD=10000
MAX_QUERY_LENGTH=200

6. Archivo .gitignore
# Entorno virtual
venv/
.env

# Datos y modelos
data/processed/
*.pkl
*.index

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
__pycache__/
*.py[cod]

# Sistema operativo
.DS_Store
Thumbs.db

7. Instalación de Dependencias
Con el entorno virtual activado:
# Instalar dependencias principales
pip install -r requirements.txt

# Instalar dependencias de desarrollo (opcional)
pip install -r requirements_dev.txt

8. Configuración Inicial del Proyecto
Crea un script setup.py en la raíz para inicialización:
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


9. Ejecución del Proyecto
# Primero configura la estructura (solo primera vez)
python setup.py

# Luego inicia el sistema
python main.py --ui  # Para interfaz gráfica
# o
python main.py       # Para línea de comandos


10. Ayudas

python -m pip install --upgrade pip


Para que puedas instalar numpy==1.26.4 (y otras librerías científicas que requieren compilación en Windows), necesitas instalar Microsoft C++ Build Tools, con los siguientes componentes específicos:

✅ Pasos exactos para instalar los C++ Build Tools correctos
Descarga el instalador desde:
👉 https://visualstudio.microsoft.com/visual-cpp-build-tools/

Ejecuta el instalador y selecciona:

🔧 Workload (caja grande a la izquierda):
✅ "Desarrollo de escritorio con C++" (Desktop development with C++)

En la parte derecha (componentes individuales), asegúrate de que estén seleccionados:

✅ MSVC v143 - VS 2022 C++ x64/x86 build tools

✅ Windows 10 SDK (versión más reciente)

✅ C++ CMake tools for Windows

✅ C++ ATL for v143 build tools (x86 & x64) (opcional pero útil)

✅ C++ CLI support (opcional)

✅ C++/WinRT (opcional)

Dale clic a "Instalar" (puede tardar unos minutos).

📌 Verifica que esté funcionando
Después de la instalación, reinicia la terminal, activa tu entorno virtual, y prueba:
pip install numpy==1.26.4
Si aún quieres evitar compilar, sigue siendo más rápido usar:
pip install numpy==1.26.4 --only-binary=:all:
