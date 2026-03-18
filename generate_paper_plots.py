#!/usr/bin/env python3
# generate_paper_plots.py
"""
generar_figuras.py
==================
Genera todas las figuras del paper con datos reales del proyecto.
Ejecutar desde la raiz del proyecto:
    python generar_figuras.py

Requiere: matplotlib, numpy
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

os.makedirs("figures", exist_ok=True)

# ── Paleta de colores IEEE-friendly (escala de grises + azul) ─────────────────
C_BASELINE  = "#4A4A4A"   # gris oscuro
C_NER       = "#2196F3"   # azul
C_REWARD    = "#1565C0"   # azul oscuro
C_PPO       = "#78909C"   # gris azulado
C_HYBRID    = "#0D47A1"   # azul muy oscuro
COLORS = [C_BASELINE, C_NER, C_REWARD, C_PPO, C_HYBRID]

METHODS     = ["Baseline\n(FAISS)", "NER-\nEnhanced", "Reward-\nOnly", "RLHF\n(PPO)", "Full\nHybrid"]
METHODS_SHORT = ["Baseline", "NER", "Reward-Only", "RLHF", "Full Hybrid"]

# ── Datos reales de la última evaluación (17/03/2026) ─────────────────────────
NDCG   = [0.8497, 0.8334, 0.8817, 0.8350, 0.8817]
RECALL = [1.0000, 0.9214, 1.0000, 1.0000, 1.0000]
MRR    = [0.8000, 0.8778, 0.9167, 0.8444, 0.9167]
MAP    = [0.7437, 0.7175, 0.7762, 0.7115, 0.7762]
DELTA  = [0.0,   -1.9,   +3.8,   -1.7,   +3.8]

# ── Evolución histórica del reward model ──────────────────────────────────────
PREF_HIST  = [31, 59, 135]
NDCG_HIST  = [0.800, 0.823, 0.8817]   # Reward-Only nDCG en cada iteración
VACC_HIST  = [0.944, 0.957, 0.909]    # val_accuracy reward model

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Comparativa nDCG@10 por método (barras horizontales)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 3.5))

y = np.arange(len(METHODS))
bars = ax.barh(y, NDCG, color=COLORS, height=0.55, edgecolor='white', linewidth=0.5)

# Línea de referencia baseline
ax.axvline(NDCG[0], color=C_BASELINE, linestyle='--', linewidth=1.2, alpha=0.7, label='Baseline')

# Etiquetas de valor
for i, (bar, val, delta) in enumerate(zip(bars, NDCG, DELTA)):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha='left', fontsize=8.5, fontweight='bold')
    if delta != 0:
        sign = '+' if delta > 0 else ''
        color_d = '#1565C0' if delta > 0 else '#B71C1C'
        ax.text(val + 0.0115, bar.get_y() + bar.get_height()/2,
                f'({sign}{delta:.1f}%)', va='center', ha='left',
                fontsize=7.5, color=color_d)

ax.set_yticks(y)
ax.set_yticklabels(METHODS, fontsize=9)
ax.set_xlabel('nDCG@10', fontsize=9)
ax.set_xlim(0.70, 0.945)
ax.set_title('Comparativa nDCG@10 por método de ranking', fontsize=10, fontweight='bold', pad=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, linestyle=':')
plt.tight_layout()
plt.savefig('figures/fig_ndcg_comparativa.pdf', bbox_inches='tight', dpi=150)
plt.savefig('figures/fig_ndcg_comparativa.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ fig_ndcg_comparativa")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Tabla de métricas completa (4 métricas × 5 métodos)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)

metrics_data = {
    'nDCG@10':    NDCG,
    'Recall@10':  RECALL,
    'MRR':        MRR,
    'MAP@10':     MAP,
}

for ax, (metric_name, values) in zip(axes, metrics_data.items()):
    bars = ax.bar(np.arange(5), values, color=COLORS, edgecolor='white', linewidth=0.5)
    ax.set_title(metric_name, fontsize=9, fontweight='bold')
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(["BL", "NER", "RO", "PPO", "FH"], fontsize=8)
    ax.set_ylim(0.65, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

axes[0].set_ylabel('Score', fontsize=9)

legend_patches = [mpatches.Patch(color=c, label=m)
                  for c, m in zip(COLORS, METHODS_SHORT)]
fig.legend(handles=legend_patches, loc='lower center', ncol=5,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.suptitle('Métricas de evaluación por método (test set, n=15 queries)',
             fontsize=10, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figures/fig_metricas_completas.pdf', bbox_inches='tight', dpi=150)
plt.savefig('figures/fig_metricas_completas.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ fig_metricas_completas")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Evolución Reward-Only nDCG con preferencias
# ══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(6, 3.5))

ax2 = ax1.twinx()

line1, = ax1.plot(PREF_HIST, NDCG_HIST, 'o-', color=C_REWARD,
                  linewidth=2, markersize=7, label='Reward-Only nDCG@10')
ax1.axhline(NDCG[0], color=C_BASELINE, linestyle='--', linewidth=1.2,
            alpha=0.7, label=f'Baseline ({NDCG[0]:.4f})')

line2, = ax2.plot(PREF_HIST, VACC_HIST, 's--', color='#43A047',
                  linewidth=1.8, markersize=6, label='Reward val_acc')

# Anotaciones
for x, y_val in zip(PREF_HIST, NDCG_HIST):
    ax1.annotate(f'{y_val:.4f}', (x, y_val), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=8.5, color=C_REWARD)

ax1.set_xlabel('Número de preferencias A/B recolectadas', fontsize=9)
ax1.set_ylabel('nDCG@10 (Reward-Only)', fontsize=9, color=C_REWARD)
ax2.set_ylabel('val_accuracy (Reward Model)', fontsize=9, color='#43A047')
ax1.set_ylim(0.76, 0.92)
ax2.set_ylim(0.88, 0.97)
ax1.set_xticks(PREF_HIST)
ax1.set_title('Evolución del sistema RLHF con datos de preferencia', fontsize=10,
              fontweight='bold', pad=8)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.grid(axis='y', alpha=0.3, linestyle=':')

lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines + [ax1.get_lines()[1]], labels + [f'Baseline ({NDCG[0]:.4f})'],
           fontsize=8, loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig('figures/fig_evolucion_rlhf.pdf', bbox_inches='tight', dpi=150)
plt.savefig('figures/fig_evolucion_rlhf.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ fig_evolucion_rlhf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 4 — Arquitectura del sistema (diagrama de flujo SVG-style)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color='#E3F2FD', edgecolor='#1565C0',
             fontsize=8, bold=False):
    rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                          facecolor=color, edgecolor=edgecolor, linewidth=1.5,
                          zorder=2)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, zorder=3, wrap=True,
            multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#455A64',
                                lw=1.5), zorder=1)

# Datos de entrada
draw_box(ax, 1.0, 4.2, 1.6, 0.6, 'Amazon\nProducts\n(9,999)', '#FFF9C4', '#F57F17', 7)
draw_box(ax, 1.0, 3.0, 1.6, 0.6, 'Interacciones\nUsuario\nA/B', '#FFF9C4', '#F57F17', 7)

# Procesamiento
draw_box(ax, 3.2, 4.2, 1.8, 0.6, 'Canonicalización\n+ Embeddings\n(MiniLM-L6-v2)', '#E3F2FD', '#1565C0', 7)
draw_box(ax, 3.2, 3.0, 1.8, 0.6, 'NER Extractor\n(DeBERTa\nnli-v3-small)', '#E8F5E9', '#2E7D32', 7)
draw_box(ax, 3.2, 1.8, 1.8, 0.6, 'Reward Model\n(MLP 435K\nval_acc=0.909)', '#FCE4EC', '#880E4F', 7)

# Índices y modelos
draw_box(ax, 5.7, 4.2, 1.8, 0.6, 'FAISS Index\n(384-dim\nAVX2)', '#E3F2FD', '#1565C0', 7)
draw_box(ax, 5.7, 3.0, 1.8, 0.6, 'NER Cache\n(7,202/9,999\nproductos)', '#E8F5E9', '#2E7D32', 7)
draw_box(ax, 5.7, 1.8, 1.8, 0.6, 'PPO Policy\n(Transformer\nheads=4)', '#FCE4EC', '#880E4F', 7)

# Métodos de ranking
draw_box(ax, 8.3, 4.5, 1.6, 0.5, 'Baseline\nnDCG=0.8497', C_BASELINE, '#212121', 7)
draw_box(ax, 8.3, 3.8, 1.6, 0.5, 'NER-Enhanced\nnDCG=0.8334', '#1976D2', '#0D47A1', 7, False)
draw_box(ax, 8.3, 3.1, 1.6, 0.5, 'Reward-Only\nnDCG=0.8817 ★', '#1565C0', '#0D47A1', 7, True)
draw_box(ax, 8.3, 2.4, 1.6, 0.5, 'RLHF (PPO)\nnDCG=0.8350', '#78909C', '#37474F', 7)
draw_box(ax, 8.3, 1.7, 1.6, 0.5, 'Full Hybrid\nnDCG=0.8817', '#0D47A1', '#01579B', 7, True)

# Flechas
draw_arrow(ax, 1.8, 4.2, 2.3, 4.2)
draw_arrow(ax, 1.8, 3.0, 2.3, 3.0)
draw_arrow(ax, 1.8, 3.0, 2.3, 1.8)
draw_arrow(ax, 4.1, 4.2, 4.8, 4.2)
draw_arrow(ax, 4.1, 3.0, 4.8, 3.0)
draw_arrow(ax, 4.1, 1.8, 4.8, 1.8)
draw_arrow(ax, 6.6, 4.2, 7.5, 4.5)
draw_arrow(ax, 6.6, 4.2, 7.5, 3.8)
draw_arrow(ax, 6.6, 3.0, 7.5, 3.1)
draw_arrow(ax, 6.6, 1.8, 7.5, 2.4)
draw_arrow(ax, 6.6, 4.2, 7.5, 1.7)
draw_arrow(ax, 6.6, 3.0, 7.5, 1.7)
draw_arrow(ax, 6.6, 1.8, 7.5, 1.7)

# Labels de secciones
ax.text(3.2, 4.85, 'Procesamiento', ha='center', fontsize=8,
        fontstyle='italic', color='#546E7A')
ax.text(5.7, 4.85, 'Índices y Modelos', ha='center', fontsize=8,
        fontstyle='italic', color='#546E7A')
ax.text(8.3, 5.0, 'Métodos de Ranking', ha='center', fontsize=8,
        fontstyle='italic', color='#546E7A')

ax.set_title('Arquitectura del sistema híbrido FAISS+NER+RLHF',
             fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('figures/fig_arquitectura.pdf', bbox_inches='tight', dpi=150)
plt.savefig('figures/fig_arquitectura.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ fig_arquitectura")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 5 — Ciclo RLHF (diagrama circular)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')
ax.set_aspect('equal')

steps = [
    (0,  1.0, 'Queries de\nTrain\n(23 queries)', '#FFF9C4', '#F57F17'),
    (0.95, 0.31, 'Sesión A/B\nInteractiva\n(135 pref.)', '#E3F2FD', '#1565C0'),
    (0.59, -0.81, 'Entrenamiento\nReward Model\n(val_acc=0.909)', '#FCE4EC', '#880E4F'),
    (-0.59, -0.81, 'PPO Training\n(50q × 10 epochs\nreward>0)', '#E8F5E9', '#2E7D32'),
    (-0.95, 0.31, 'Evaluación\nnDCG@10\n(n=15)', '#F3E5F5', '#6A1B9A'),
]

for (x, y, label, fc, ec) in steps:
    circle = plt.Circle((x, y), 0.35, color=fc, ec=ec, linewidth=2, zorder=2)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=7.5,
            fontweight='bold', zorder=3, multialignment='center')

# Flechas circulares
angles = [90, 90-72, 90-144, 90-216, 90-288]
for i in range(5):
    a1 = np.radians(angles[i] - 25)
    a2 = np.radians(angles[(i+1) % 5] + 25)
    x1, y1 = 0.85 * np.cos(a1), 0.85 * np.sin(a1)
    x2, y2 = 0.85 * np.cos(a2), 0.85 * np.sin(a2)
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#455A64',
                                lw=1.8, connectionstyle='arc3,rad=0.3'))

ax.text(0, 0, 'Ciclo\nRLHF', ha='center', va='center', fontsize=10,
        fontweight='bold', color='#37474F')

ax.set_title('Ciclo de entrenamiento RLHF implementado', fontsize=10,
             fontweight='bold', pad=5)

plt.tight_layout()
plt.savefig('figures/fig_ciclo_rlhf.pdf', bbox_inches='tight', dpi=150)
plt.savefig('figures/fig_ciclo_rlhf.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ fig_ciclo_rlhf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 6 — NER: intents detectados por DeBERTa (ejemplo)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis('off')

queries_ex = [
    "survival horror\nvideogames",
    "mario games",
    "smash bros",
    "need for speed",
    "playstation games",
]
intents_ex = [
    "genre: [horror, survival]\nfranchise: [Resident Evil]\nfeatures: [story-driven]",
    "genre: [platformer, metroidvania]\nfranchise: [Mario]",
    "platform: [Nintendo Switch]\ngenre: [action, fighting]\nfranchise: [Smash Bros]",
    "genre: [action, racing, sports]\nfranchise: [Need for Speed]",
    "platform: [PlayStation 4]\nfeatures: [online multiplayer]",
]

table_data = list(zip(queries_ex, intents_ex))
col_labels = ['Query', 'Intent detectado por DeBERTa NLI']
col_widths = [0.22, 0.78]

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='left',
    loc='center',
    colWidths=col_widths,
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2.5)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#E3F2FD')
    else:
        cell.set_facecolor('#FAFAFA')
    cell.set_edgecolor('#CFD8DC')

ax.set_title('Ejemplos de detección de intent con DeBERTa zero-shot NLI',
             fontsize=10, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('figures/fig_ner_ejemplos.pdf', bbox_inches='tight', dpi=150)
plt.savefig('figures/fig_ner_ejemplos.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ fig_ner_ejemplos")

print("\nTodas las figuras generadas en ./figures/")
print("Archivos: fig_ndcg_comparativa, fig_metricas_completas,")
print("          fig_evolucion_rlhf, fig_arquitectura,")
print("          fig_ciclo_rlhf, fig_ner_ejemplos")
print("Formatos: .pdf (para LaTeX) y .png (para preview)")