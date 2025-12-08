#!/usr/bin/env python3
"""
generar_grafica_rlhf.py - Gráfica de evolución del aprendizaje RLHF
"""
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

def generate_rlhf_evolution_chart():
    # Simular datos de entrenamiento (reemplaza con datos reales)
    sessions = list(range(1, 21))
    
    # Scores de recomendación
    recommendation_scores = np.array([
        0.45, 0.48, 0.52, 0.55, 0.58, 0.62, 0.65, 0.68, 0.71, 0.73,
        0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.80, 0.81, 0.81, 0.82
    ])
    
    # Tasa de feedback positivo
    positive_feedback = np.array([
        0.30, 0.32, 0.35, 0.38, 0.42, 0.45, 0.48, 0.52, 0.55, 0.58,
        0.60, 0.62, 0.64, 0.66, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73
    ])
    
    # Pérdida del modelo
    training_loss = np.array([
        0.85, 0.78, 0.72, 0.67, 0.63, 0.59, 0.56, 0.53, 0.50, 0.48,
        0.46, 0.44, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.37, 0.36
    ])
    
    # Configurar gráfica
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfica superior: Score y feedback
    ax1.plot(sessions, recommendation_scores, 'o-', linewidth=2.5, markersize=6,
             label='Score de Recomendación', color='#2E86AB', markerfacecolor='white')
    ax1.plot(sessions, positive_feedback, 's--', linewidth=2.5, markersize=6,
             label='Feedback Positivo (%)', color='#A23B72', markerfacecolor='white')
    
    ax1.set_xlabel('Sesiones de Entrenamiento', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score / % Feedback', fontsize=12, fontweight='bold')
    ax1.set_title('Evolución del Aprendizaje RLHF', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(sessions[::2])
    ax1.set_ylim(0.2, 0.9)
    
    # Línea de umbral aceptable
    ax1.axhline(y=0.7, color='#FF6B6B', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(1, 0.71, 'Umbral aceptable', color='#FF6B6B', fontsize=10)
    
    # Gráfica inferior: Pérdida de entrenamiento
    ax2.plot(sessions, training_loss, '^-', linewidth=2.5, markersize=8,
             label='Pérdida de Entrenamiento', color='#FF9F1C', markerfacecolor='white')
    
    # Área sombreada para convergencia
    ax2.fill_between(sessions[10:], training_loss[10:] - 0.02, 
                     training_loss[10:] + 0.02, alpha=0.2, color='#FF9F1C')
    
    ax2.set_xlabel('Sesiones de Entrenamiento', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pérdida', fontsize=12, fontweight='bold')
    ax2.set_title('Convergencia del Modelo RLHF', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(sessions[::2])
    ax2.set_ylim(0.3, 0.9)
    
    # Añadir anotaciones
    ax1.annotate('Mejora inicial rápida', xy=(3, 0.55), xytext=(6, 0.4),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    ax1.annotate('Convergencia estable', xy=(15, 0.80), xytext=(10, 0.85),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    ax2.annotate('Zona de convergencia', xy=(15, 0.39), xytext=(10, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar
    output_path = Path("graficas") / "evolucion_rlhf.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Gráfica guardada en: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_rlhf_evolution_chart()