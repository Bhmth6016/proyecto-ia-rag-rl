import json
import os

# Ruta relativa al archivo de datos
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'amazon_products.json')

def cargar_productos():
    """Carga el dataset de productos desde el archivo JSON."""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        productos = json.load(f)
    return productos

def buscar_productos(texto_busqueda, productos, max_resultados=5):
    """
    Busca productos que contengan el texto de búsqueda en nombre o descripción.
    Retorna una lista con los productos encontrados (hasta max_resultados).
    """
    texto = texto_busqueda.lower()
    resultados = []
    for producto in productos:
        nombre = producto['nombre'].lower()
        descripcion = producto['descripcion'].lower()
        if texto in nombre or texto in descripcion:
            resultados.append(producto)
        if len(resultados) >= max_resultados:
            break
    return resultados

# Prueba rápida
if __name__ == "__main__":
    productos = cargar_productos()
    busqueda = "audífonos"
    encontrados = buscar_productos(busqueda, productos)
    for p in encontrados:
        print(f"{p['nombre']} - ${p['precio']}")
