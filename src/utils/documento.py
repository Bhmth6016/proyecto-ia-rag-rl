from langchain.schema import Document

def producto_a_documento(producto):
    detalles = producto.get("details", {})
    texto = f"Título: {producto.get('title', '')}.\n"
    texto += f"Categoría principal: {producto.get('main_category', '')}.\n"
    texto += f"Precio: ${producto.get('price', 'No disponible')}.\n"
    texto += f"Valoración promedio: {producto.get('average_rating', 'N/A')} estrellas en {producto.get('rating_number', 'N/A')} reseñas.\n"

    if 'features' in producto:
        texto += "Características: " + "; ".join(producto['features']) + "\n"

    if 'description' in producto and isinstance(producto['description'], list):
        texto += "Descripción: " + " ".join(producto['description']) + "\n"

    texto += f"Detalles técnicos: {', '.join([f'{k}: {v}' for k, v in detalles.items()])}\n"

    return Document(page_content=texto, metadata={"title": producto.get("title", "Sin título")})