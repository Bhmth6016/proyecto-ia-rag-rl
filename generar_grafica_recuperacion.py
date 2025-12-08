#!/usr/bin/env python3
"""
generar_grafica_recuperacion.py - Gráfica comparativa RAG vs baseline
"""
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Datos de ejemplo (reemplaza con tus datos reales)
def generate_retrieval_chart():
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top-K Accuracy
    k_values = [1, 3, 5, 10]
    rag_accuracy = [0.35, 0.62, 0.78, 0.88]  # RAG con embeddings ML
    baseline_accuracy = [0.25, 0.45, 0.58, 0.68]  # TF-IDF tradicional
    
    ax1.plot(k_values, rag_accuracy, 'o-', linewidth=2.5, markersize=8, 
             label='RAG con ML', color='#2E86AB', markerfacecolor='white')
    ax1.plot(k_values, baseline_accuracy, 's--', linewidth=2.5, markersize=8,
             label='TF-IDF Baseline', color='#A23B72', markerfacecolor='white')
    
    ax1.set_xlabel('Top-K', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precisión', fontsize=12, fontweight='bold')
    ax1.set_title('Precisión en Recuperación Semántica', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Tiempo de respuesta
    models = ['RAG Básico', 'RAG + NLP', 'RAG + ML', 'RAG Completo']
    avg_times = [28.5, 32.1, 45.3, 52.8]  # ms
    std_dev = [3.2, 4.1, 5.8, 6.5]
    
    colors = ['#4ECDC4', '#FF6B6B', '#FFD166', '#06D6A0']
    bars = ax2.bar(models, avg_times, yerr=std_dev, capsize=8, 
                   color=colors, edgecolor='black', linewidth=1.2)
    
    # Añadir etiquetas de valor
    for bar, time in zip(bars, avg_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Configuración del Sistema', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Tiempo (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Tiempo de Respuesta por Configuración', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Guardar
    output_path = Path("graficas") / "recuperacion_semantica.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Gráfica guardada en: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_retrieval_chart()