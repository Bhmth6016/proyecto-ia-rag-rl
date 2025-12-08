#!/usr/bin/env python3
"""
generar_grafica_comparativa.py - Gráfica de comparación entre configuraciones
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_comparative_score_chart():
    # Configurar estilo profesional
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Datos de los 4 puntos evaluados
    configs = ['Punto 1\nBásico', 'Punto 2\n+NLP', 'Punto 3\n+ML', 'Punto 4\nCompleto']
    
    # Métricas por configuración
    metrics = {
        'Precisión': [0.58, 0.67, 0.75, 0.82],
        'Tiempo (ms)': [28.5, 32.1, 45.3, 52.8],
        'Recall@5': [0.62, 0.71, 0.78, 0.85],
        'User Satisfaction': [0.65, 0.72, 0.78, 0.84]
    }
    
    colors = ['#4ECDC4', '#FF6B6B', '#FFD166', '#06D6A0']
    
    # Gráfica 1: Comparación general
    x = np.arange(len(configs))
    width = 0.2
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - 1.5) * width
        bars = ax1.bar(x + offset, values, width, label=metric_name, 
                      color=colors[i], edgecolor='black', linewidth=0.8)
        
        # Añadir etiquetas
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Configuración del Sistema', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Comparativa Completa de Métricas', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfica 2: Precisión vs Tiempo (trade-off)
    precision = metrics['Precisión']
    time_ms = metrics['Tiempo (ms)']
    
    scatter = ax2.scatter(time_ms, precision, s=200, c=colors, edgecolors='black', linewidth=1.5)
    
    # Conectar puntos
    ax2.plot(time_ms, precision, '--', color='gray', alpha=0.5)
    
    # Anotar cada punto
    for i, (t, p, config) in enumerate(zip(time_ms, precision, configs)):
        ax2.annotate(config, (t, p), xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold', alpha=0.8)
    
    ax2.set_xlabel('Tiempo de Respuesta (ms)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precisión', fontsize=12, fontweight='bold')
    ax2.set_title('Trade-off: Precisión vs Tiempo', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Mejora porcentual
    baseline = metrics['Precisión'][0]
    improvements = [(v - baseline) / baseline * 100 for v in metrics['Precisión']]
    
    bars3 = ax3.bar(configs, improvements, color=colors, edgecolor='black', linewidth=1.2)
    
    # Añadir etiquetas
    for bar, imp in zip(bars3, improvements):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'+{imp:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xlabel('Configuración', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mejora % vs Baseline', fontsize=12, fontweight='bold')
    ax3.set_title('Mejora Relativa en Precisión', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linewidth=1)
    
    # Gráfica 4: Distribución por tipo de consulta
    query_types = ['Simple', 'Compleja', 'Técnica']
    config_scores = np.array([
        [0.65, 0.55, 0.52],  # Punto 1
        [0.72, 0.68, 0.61],  # Punto 2
        [0.78, 0.76, 0.71],  # Punto 3
        [0.85, 0.82, 0.79]   # Punto 4
    ])
    
    x = np.arange(len(query_types))
    width = 0.2
    
    for i in range(len(configs)):
        ax4.bar(x + (i-1.5)*width, config_scores[i], width, label=configs[i],
               color=colors[i], edgecolor='black', linewidth=0.8)
    
    ax4.set_xlabel('Tipo de Consulta', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('Desempeño por Tipo de Consulta', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(query_types)
    ax4.legend(title='Configuración', loc='upper left', bbox_to_anchor=(1, 1))
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Análisis Comparativo de Configuraciones del Sistema', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Guardar
    output_path = Path("graficas") / "comparativa_scores.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Gráfica guardada en: {output_path}")
    
    # Generar resumen
    print("\n📊 RESUMEN ESTADÍSTICO:")
    print("-" * 50)
    print(f"{'Configuración':<20} {'Precisión':<10} {'Tiempo (ms)':<12} {'Mejora %'}")
    print("-" * 50)
    for i, config in enumerate(configs):
        print(f"{config:<20} {metrics['Precisión'][i]:<10.3f} {metrics['Tiempo (ms)'][i]:<12.1f} "
              f"+{improvements[i]:.1f}%")
    
    return output_path

if __name__ == "__main__":
    generate_comparative_score_chart()