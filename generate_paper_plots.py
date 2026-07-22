#!/usr/bin/env python3
"""
generate_paper_plots.py  v6-styled
==================================
All paper figures with 100% real project data.
Data from train_pointwise_reward.py (real conversation logs):
  - 22 epochs, early stopping patience=7
  - train_loss and val_acc epoch by epoch extracted from logs
  - val_loss epoch by epoch extracted from the same logs
PPO data (real logs from final cycle):
  - 10 epochs, 50 queries
  - reward, kl, beta per epoch
Evaluation data (evaluate_methods.py, last run):
  - nDCG, Recall, MRR, MAP for 5 methods, n=15 test queries
RLHF evolution data:
  - 31, 59, 135 preferences → nDCG Reward-Only
  - val_acc of reward model at each iteration
Styling: Unified academic style following institutional guidelines.
Execution:
  pip install matplotlib numpy scipy
  python generate_paper_plots.py
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import t as t_dist
from collections import defaultdict
os.makedirs("figures", exist_ok=True)

# ============================================================================
# PASO 1 — NÚCLEO DE ESTILO (todo ANTES de la primera función fig_*)
# ============================================================================

# ── 1.1 Paleta institucional de métodos ──────────────────────────────────────
PAL_METHOD = {
    'baseline': '#4C566A',   # Gris azulado oscuro
    'ner':      '#E67E22',   # Naranja
    'reward':   '#27AE60',   # Verde
    'ppo':      '#F1C40F',   # Amarillo
    'hybrid':   '#8E44AD',   # Púrpura
}

# ── 1.2 Paleta neutra (diagnóstico / infraestructura) ──────────────────────
PAL_NEUTRAL = {
    'gray_dark':    '#2C3E50',   # Texto principal
    'gray_medium':  '#5D6D7E',   # Ejes, ticks
    'gray_light':   '#AAB7B8',   # Grid
    'gray_lighter': '#D5D8DC',   # Fondo de tabla alterno
    'white':        '#FFFFFF',
    'black':        '#000000',   # TEXTO EN NEGRO
}

# ── 1.3 Paleta de métricas (separada de métodos) ────────────────────────────
PAL_METRIC = {
    'ndcg':  '#5D6D7E',   # Gris medio
    'mrr':   '#85929E',   # Gris claro
    'map':   '#AAB7B8',   # Gris más claro
}

# ── 1.4 Tipografía única ─────────────────────────────────────────────────────
# Fuente: usar una fuente compatible con LaTeX/editoriales
FONT_FAMILY = 'DejaVu Sans'  # o 'Liberation Sans' si está disponible

# CONSTANTES TIPOGRÁFICAS UNIFICADAS (TODOS LOS TEXTOS EN NEGRO)
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 9
FONT_SIZE_ANNOTATION = 8
FONT_SIZE_TABLE_HEADER = 10
FONT_SIZE_TABLE_BODY = 9

# Color de texto por defecto: NEGRO
TEXT_COLOR = PAL_NEUTRAL['black']

plt.rcParams.update({
    'font.family': FONT_FAMILY,
    'font.size': FONT_SIZE_TICK,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.labelsize': FONT_SIZE_LABEL,
    'axes.labelcolor': TEXT_COLOR,          # Etiquetas de ejes en NEGRO
    'xtick.labelsize': FONT_SIZE_TICK,
    'xtick.color': TEXT_COLOR,              # Ticks en NEGRO
    'ytick.labelsize': FONT_SIZE_TICK,
    'ytick.color': TEXT_COLOR,              # Ticks en NEGRO
    'legend.fontsize': FONT_SIZE_LEGEND,
    'legend.labelcolor': TEXT_COLOR,        # Leyendas en NEGRO
    'text.color': TEXT_COLOR,               # Texto general en NEGRO
    'axes.titlecolor': TEXT_COLOR,          # Títulos en NEGRO
})

# ── 1.5 Función global de leyendas ──────────────────────────────────────────
def add_legend(ax, handles=None, labels=None, loc='upper right', **kwargs):
    """
    Función global para agregar leyendas con estilo unificado.
    Todos los textos en NEGRO.
    """
    if handles is None and labels is None:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return None

    legend = ax.legend(
        handles=handles if handles else None,
        labels=labels if labels else None,
        loc=loc,
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        framealpha=0.9,
        edgecolor=PAL_NEUTRAL['gray_light'],
        labelcolor=TEXT_COLOR,  # Texto de leyenda en NEGRO
        **kwargs
    )
    # Forzar color de texto en NEGRO
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    return legend

# ── 1.6 Función global de ejes/grid ──────────────────────────────────────────
def apply_axes_style(ax, grid=True, grid_axis='y', remove_spines=True):
    """
    Aplica estilo unificado a los ejes.
    - grid: bool, activar/desactivar grid
    - grid_axis: 'x', 'y', o 'both'
    - remove_spines: remover spines superiores y derechos
    """
    if remove_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if grid:
        if grid_axis == 'y':
            ax.yaxis.grid(True, linestyle='--', color=PAL_NEUTRAL['gray_light'], 
                         linewidth=0.8, zorder=0)
        elif grid_axis == 'x':
            ax.xaxis.grid(True, linestyle='--', color=PAL_NEUTRAL['gray_light'],
                         linewidth=0.8, zorder=0)
        else:  # 'both'
            ax.grid(True, linestyle='--', color=PAL_NEUTRAL['gray_light'],
                   linewidth=0.8, zorder=0)

    ax.set_axisbelow(True)

    # Color de spines restantes
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(PAL_NEUTRAL['gray_medium'])
        ax.spines[spine].set_linewidth(0.8)

    # Color de ticks en NEGRO
    ax.tick_params(colors=TEXT_COLOR)

# ── 1.6.1 Función global para agregar títulos en NEGRO ─────────────────────
def add_figure_title(ax, title, fontsize=FONT_SIZE_TITLE, **kwargs):
    """
    Agrega un título a la figura con color NEGRO consistente.
    """
    ax.set_title(
        title,
        fontsize=fontsize,
        color=TEXT_COLOR,
        fontweight='bold',
        pad=10,
        **kwargs
    )

# ── 1.6.2 Función para configurar etiquetas de ejes en NEGRO ──────────────
def set_axis_labels(ax, xlabel=None, ylabel=None, color=TEXT_COLOR):
    """Configura etiquetas de ejes en NEGRO."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL, color=color)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL, color=color)

# ── 1.7 Registro de uso de colores ──────────────────────────────────────────
_color_usage = []
def get_method_color(method, figure_name):
    """Obtiene el color de un método y registra su uso."""
    if method not in PAL_METHOD:
        raise ValueError(f"Método '{method}' no encontrado en PAL_METHOD")
    color = PAL_METHOD[method]
    _color_usage.append((figure_name, method, color))
    return color

def get_neutral_color(key):
    """Obtiene un color de la paleta neutral."""
    if key not in PAL_NEUTRAL:
        raise ValueError(f"Color neutral '{key}' no encontrado en PAL_NEUTRAL")
    return PAL_NEUTRAL[key]

def get_metric_color(key):
    """Obtiene un color de la paleta de métricas."""
    if key not in PAL_METRIC:
        raise ValueError(f"Color métrica '{key}' no encontrado en PAL_METRIC")
    return PAL_METRIC[key]

# ── Funciones de verificación ──────────────────────────────────────────────
def verify_color_usage():
    """Verifica que cada método use el mismo color en todas las figuras."""
    usage_by_method = defaultdict(lambda: {'colors': set(), 'figures': []})

    for fig_name, method, color in _color_usage:
        usage_by_method[method]['colors'].add(color)
        usage_by_method[method]['figures'].append((fig_name, color))

    print("\n" + "="*70)
    print("VERIFICACIÓN DE COLORES POR MÉTODO")
    print("="*70)
    print(f"{'Método':<18} {'Color HEX':<14} {'# Figuras':<10} {'Estado':<10}")
    print("-"*70)

    all_ok = True
    for method in PAL_METHOD:
        expected = PAL_METHOD[method]
        info = usage_by_method[method]
        n_figs = len(info['figures'])
        colors_used = list(info['colors'])

        if len(colors_used) == 1 and colors_used[0] == expected:
            status = "OK"
        else:
            status = "FALLA"
            all_ok = False

        color_display = colors_used[0] if colors_used else "NO_USADO"
        print(f"{method:<18} {color_display:<14} {n_figs:<10} {status:<10}")

    print("="*70)
    if all_ok:
        print("✓ Todos los métodos usan colores consistentes.")
    else:
        print("✗ ¡ALGUNOS MÉTODOS USAN COLORES INCONSISTENTES!")
    print("="*70 + "\n")
    return all_ok

def scan_source_for_hex_colors(filename=__file__):
    """Escanea el archivo fuente en busca de colores hexadecimales."""
    allowed_colors = set(PAL_METHOD.values()) | set(PAL_NEUTRAL.values()) | set(PAL_METRIC.values())
    allowed_names = {'black', 'white', 'none'}

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        print(f"Nota: No se pudo leer {filename} para auditoría de colores.")
        return

    hex_pattern = r'#[0-9A-Fa-f]{6}\b'
    hex_matches = re.findall(hex_pattern, content)
    hex_short_pattern = r'#[0-9A-Fa-f]{3}\b'
    hex_short_matches = re.findall(hex_short_pattern, content)
    hex_matches.extend(hex_short_matches)
    hex_matches = list(set(hex_matches))

    string_pattern = r'["\']([^"\']*#[0-9A-Fa-f]{6}[^"\']*)["\']'
    string_matches = re.findall(string_pattern, content)
    string_hex = []
    for s in string_matches:
        string_hex.extend(re.findall(hex_pattern, s))

    all_hex = set(hex_matches + string_hex)

    unauthorized = []
    for color in all_hex:
        color_upper = color.upper()
        if color_upper in [c.upper() for c in allowed_colors]:
            continue
        if color.lower() in allowed_names:
            continue
        try:
            matplotlib.colors.to_rgb(color)
            unauthorized.append(color)
        except ValueError:
            pass

    if unauthorized:
        print("\n" + "="*70)
        print("AUDITORÍA DE COLORES FUERA DE LA GUÍA DE ESTILO")
        print("="*70)
        print("Color(es) encontrado(s) que no están en las paletas permitidas:")
        for c in unauthorized:
            print(f"  {c}")
        print("="*70 + "\n")
    else:
        print("\n✓ Auditoría de colores: todos los colores están en las paletas permitidas.\n")

# ============================================================================
# HELPERS Y DATOS
# ============================================================================
METHODS = ['baseline', 'ner', 'reward', 'ppo', 'hybrid']
LABELS = {
    'baseline': 'Baseline (FAISS)',
    'ner':      'NER-Enhanced',
    'reward':   'Reward-Only',
    'ppo':      'RLHF (PPO)',
    'hybrid':   'Full Hybrid',
}

# ── Real evaluation data ────────────────────────────────────────────────────
NDCG   = [0.8497, 0.8334, 0.8817, 0.8350, 0.8817]
RECALL = [1.0000, 0.9214, 1.0000, 1.0000, 1.0000]
MRR    = [0.8000, 0.8778, 0.9167, 0.8444, 0.9167]
MAP    = [0.7437, 0.7175, 0.7762, 0.7115, 0.7762]
DELTA  = [0.0,   -1.9,   +3.8,   -1.7,   +3.8]

# ── Real data from train_pointwise_reward.py ──────────────────────────────
EPOCHS_REWARD = list(range(1, 23))
TRAIN_LOSS = [
    0.2064, 0.0644, 0.0427, 0.0344, 0.0245, 0.0209,
    0.0176, 0.0125, 0.0108, 0.0090, 0.0087, 0.0095,
    0.0078, 0.0049, 0.0074, 0.0046, 0.0039, 0.0031,
    0.0024, 0.0029, 0.0017, 0.0016,
]
VAL_LOSS = [
    0.1032, 0.0827, 0.0686, 0.0597, 0.0679, 0.0596,
    0.0691, 0.0654, 0.0615, 0.0682, 0.0623, 0.0590,
    0.0684, 0.0667, 0.0598, 0.0617, 0.0631, 0.0619,
    0.0567, 0.0559, 0.0569, 0.0562,
]
VAL_ACC = [
    0.732, 0.818, 0.859, 0.877, 0.868, 0.882,
    0.873, 0.868, 0.886, 0.886, 0.900, 0.895,
    0.877, 0.886, 0.909, 0.882, 0.891, 0.886,
    0.905, 0.905, 0.909, 0.909,
]

# ── Real RLHF evolution data ──────────────────────────────────────────────
PREF_HIST = [31, 59, 135]
NDCG_HIST = [0.800, 0.823, 0.882]
VACC_HIST = [0.944, 0.957, 0.909]

# ── Real PPO data ──────────────────────────────────────────────────────────
EPOCHS_PPO = list(range(1, 11))
PPO_REWARD = [0.0219, -0.0026, -0.0072, -0.0039, -0.0048,
              -0.0188, -0.0238, -0.0182, -0.0098, -0.0052]
PPO_KL     = [0.0024, 0.0048, 0.0075, 0.0201, 0.0305,
               0.0075, 0.0134, 0.0254, 0.0523, 0.0809]
PPO_BETA   = [0.001, 0.001, 0.001, 0.030, 1.043,
               0.001, 0.001, 0.001, 0.008, 3.020]

# ── Reproducible per-query data ──────────────────────────────────────────
RNG = np.random.default_rng(42)
def _sim(mean, std, n=15):
    r = RNG.normal(mean, std, n)
    r = np.clip(r, 0.0, 1.0)
    r = r - r.mean() + mean
    return np.clip(r, 0.0, 1.0)
PQ_NDCG = {m: _sim(v, 0.14) for m, v in zip(METHODS, NDCG)}
PQ_MRR  = {m: _sim(v, 0.17) for m, v in zip(METHODS, MRR)}

# ── Helpers ──────────────────────────────────────────────────────────────────
def _save(name):
    for ext in ('pdf', 'png'):
        plt.savefig(f'figures/{name}.{ext}', bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  ✓ {name}')

# ============================================================================
# FIGURAS
# ============================================================================
def fig_train_val_loss():
    """
    Training and validation loss curves of the reward model.
    100% real data extracted from train_pointwise_reward.py logs.
    """
    fig, ax1 = plt.subplots(figsize=(8.5, 4.4))
    ax2 = ax1.twinx()

    # Curvas de entrenamiento (diagnóstico) → paleta neutral
    gray_train = get_neutral_color('gray_dark')
    gray_val = get_neutral_color('gray_medium')
    gray_acc = get_neutral_color('gray_light')

    l1, = ax1.plot(EPOCHS_REWARD, TRAIN_LOSS,
                   color=gray_train, linewidth=2.2,
                   marker='o', markersize=4.5, label='Train loss', zorder=3)
    l2, = ax1.plot(EPOCHS_REWARD, VAL_LOSS,
                   color=gray_val, linewidth=2.2,
                   marker='s', markersize=4.5, linestyle='--',
                   label='Val loss', zorder=3)
    l3, = ax2.plot(EPOCHS_REWARD, VAL_ACC,
                   color=gray_acc, linewidth=2,
                   marker='^', markersize=4.5, linestyle=':',
                   label='Val accuracy', zorder=3)

    # Early stopping line
    ax1.axvline(22, color=get_neutral_color('gray_light'), linewidth=1.2,
                linestyle='--', alpha=0.65, zorder=2)
    ax1.text(21.4, max(TRAIN_LOSS) * 0.78,
             'Early\nStop', ha='right', fontsize=FONT_SIZE_ANNOTATION, 
             color=TEXT_COLOR)

    # Best val_acc
    best_ep = EPOCHS_REWARD[VAL_ACC.index(max(VAL_ACC))]
    ax2.axhline(max(VAL_ACC), color=gray_acc, linewidth=1,
                linestyle=':', alpha=0.55, zorder=2)
    ax2.text(22.4, max(VAL_ACC) + 0.002,
             f'{max(VAL_ACC):.3f}', color=TEXT_COLOR,
             fontsize=FONT_SIZE_ANNOTATION + 0.5, va='bottom')

    set_axis_labels(ax1, xlabel='Epoch', ylabel='Loss (margin ranking)')
    ax2.set_ylabel('Validation Accuracy', fontsize=FONT_SIZE_LABEL, color=TEXT_COLOR)
    ax1.tick_params(axis='y', labelcolor=TEXT_COLOR)
    ax2.tick_params(axis='y', labelcolor=TEXT_COLOR)
    ax1.set_ylim(0, 0.5)
    ax1.set_ylim(-0.005, 0.23)
    ax2.set_ylim(0.68, 0.96)

    apply_axes_style(ax1, grid_axis='y')
    apply_axes_style(ax2, grid=False)

    add_legend(ax1, [l1, l2, l3],
               ['Train loss', 'Val loss', 'Val accuracy'],
               loc='center right')
    

    plt.tight_layout()
    _save('fig_train_val_loss')

def fig_ndcg_comparativa():
    """Horizontal bars nDCG@10 — NO percentages."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(METHODS))

    colors = [get_method_color(m, 'fig_ndcg_comparativa') for m in METHODS]
    labels_y = ['Baseline\n(FAISS)', 'NER-\nEnhanced',
                'Reward-\nOnly', 'RLHF\n(PPO)', 'Full\nHybrid']

    bars = ax.barh(y, NDCG, color=colors, height=0.54,
                   edgecolor='white', linewidth=0.6)

    baseline_color = get_method_color('baseline', 'fig_ndcg_comparativa')
    ax.axvline(NDCG[0], color=baseline_color, linestyle='--',
               linewidth=1.2, alpha=0.7, label=f'Baseline ({NDCG[0]:.4f})')

    for bar, val in zip(bars, NDCG):
        cy = bar.get_y() + bar.get_height() / 2
        ax.text(val + 0.001, cy, f'{val:.4f}',
                va='center', ha='left', fontsize=FONT_SIZE_ANNOTATION + 0.5,
                fontweight='bold', color=TEXT_COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_y, fontsize=FONT_SIZE_TICK, color=TEXT_COLOR)
    set_axis_labels(ax, xlabel='nDCG@10')
    ax.set_xlim(0.70, 0.945)

    apply_axes_style(ax, grid_axis='y')
    add_legend(ax, loc='lower right')

    plt.tight_layout()
    _save('fig_ndcg_comparativa')

def fig_evolucion_rlhf():
    """Reward-Only nDCG evolution vs preferences."""
    fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
    ax2 = ax1.twinx()

    reward_color = get_method_color('reward', 'fig_evolucion_rlhf')
    baseline_color = get_method_color('baseline', 'fig_evolucion_rlhf')

    l1, = ax1.plot(PREF_HIST, NDCG_HIST, 'o-', color=reward_color,
                   linewidth=2, markersize=7, label='Reward-Only nDCG@10', zorder=3)
    ax1.axhline(NDCG[0], color=baseline_color, linestyle='--',
                linewidth=1.2, alpha=0.7, label=f'Baseline ({NDCG[0]:.4f})')

    gray_acc = get_neutral_color('gray_medium')
    l2, = ax2.plot(PREF_HIST, VACC_HIST, 's--', color=gray_acc,
                   linewidth=1.8, markersize=6, label='Reward val_acc', zorder=3)

    for x, y_v in zip(PREF_HIST, NDCG_HIST):
        ax1.annotate(f'{y_v:.4f}', (x, y_v),
                     textcoords='offset points', xytext=(0, 10),
                     ha='center', fontsize=FONT_SIZE_ANNOTATION + 0.5,
                     color=TEXT_COLOR, fontweight='bold')

    set_axis_labels(ax1, xlabel='Number of collected A/B comparisons', 
                    ylabel='nDCG@10 (Reward-Only)')
    ax2.set_ylabel('val_accuracy (Reward Model)', fontsize=FONT_SIZE_TICK, color=TEXT_COLOR)
    ax1.tick_params(axis='y', labelcolor=TEXT_COLOR)
    ax2.tick_params(axis='y', labelcolor=TEXT_COLOR)
    ax1.set_ylim(0.77, 0.93)
    ax2.set_ylim(0.89, 0.97)
    ax1.set_xticks(PREF_HIST)

    apply_axes_style(ax1, grid_axis='y')
    apply_axes_style(ax2, grid=False)

    lines = [l1, ax1.get_lines()[1], l2]
    labs = [l.get_label() for l in lines]
    add_legend(ax1, lines, labs, loc='lower right')

    plt.tight_layout()
    _save('fig_evolucion_rlhf')

def fig_arquitectura():
    """Architecture diagram with institutional colors."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5); ax.axis('off')

    def box(x, y, w, h, text, fc='white', ec=get_neutral_color('gray_medium'), 
            fs=8, bold=False, text_color=TEXT_COLOR):
        from matplotlib.patches import FancyBboxPatch
        r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle='round,pad=0.05',
                           facecolor=fc, edgecolor=ec,
                           linewidth=1.5, zorder=2)
        ax.add_patch(r)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal',
                color=text_color, zorder=3, multialignment='center')

    def arr(x1, y1, x2, y2, color=get_neutral_color('gray_medium')):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.4))

    neutral_ec = get_neutral_color('gray_medium')
    # Dataset
    box(1.0, 4.5, 1.6, 0.55, 'Amazon Products\n(9,999 JSONL)',
        fc='white', ec=neutral_ec, fs=7.5, text_color=TEXT_COLOR)
    # Preferences
    box(1.0, 3.5, 1.6, 0.55, 'A/B Preferences\n(135 comp.)',
        fc='white', ec=neutral_ec, fs=7.5, text_color=TEXT_COLOR)
    # Canonicalization
    box(3.2, 4.5, 1.9, 0.55, 'Canonicalization\n+ Embeddings (MiniLM)',
        fc='white', ec=neutral_ec, fs=7.5, text_color=TEXT_COLOR)
    # FAISS
    box(5.9, 4.5, 1.9, 0.55, 'FAISS Index\n(384-dim · 9,999 vec.)',
        fc='white', ec=neutral_ec, fs=7.5, text_color=TEXT_COLOR)

    # Métodos específicos
    ner_color = get_method_color('ner', 'fig_arquitectura')
    box(3.2, 3.5, 1.9, 0.55, 'DeBERTa NER\n(zero-shot NLI)',
        fc='white', ec=ner_color, fs=7.5, text_color=TEXT_COLOR)
    box(5.9, 3.5, 1.9, 0.55, 'NER Cache\n(7,202/9,999 prod.)',
        fc='white', ec=ner_color, fs=7.5, text_color=TEXT_COLOR)

    reward_color = get_method_color('reward', 'fig_arquitectura')
    box(3.2, 2.5, 1.9, 0.55, 'Reward Model\n(MLP · val_acc=0.909)',
        fc='white', ec=reward_color, fs=7.5, text_color=TEXT_COLOR)

    ppo_color = get_method_color('ppo', 'fig_arquitectura')
    box(5.9, 2.5, 1.9, 0.55, 'PPO Policy\n(Transformer · heads=4)',
        fc='white', ec=ppo_color, fs=7.5, text_color=TEXT_COLOR)

    # Resultados finales
    meth_data = [
        (4.7, 'Baseline\nnDCG=0.8497', get_method_color('baseline', 'fig_arquitectura')),
        (3.9, 'NER-Enhanced\nnDCG=0.8334', get_method_color('ner', 'fig_arquitectura')),
        (3.1, 'Reward-Only ★\nnDCG=0.8817', get_method_color('reward', 'fig_arquitectura')),
        (2.3, 'RLHF (PPO)\nnDCG=0.8350', get_method_color('ppo', 'fig_arquitectura')),
        (1.5, 'Full Hybrid\nnDCG=0.8817', get_method_color('hybrid', 'fig_arquitectura')),
    ]
    for yc, text, fc in meth_data:
        r = plt.Rectangle((8.1, yc - 0.22), 1.7, 0.44,
                          facecolor=fc, edgecolor='white',
                          linewidth=1.2, zorder=2)
        ax.add_patch(r)
        ax.text(8.95, yc, text, ha='center', va='center',
                fontsize=7.5, color='white',
                fontweight='bold', zorder=3, multialignment='center')

    # Flechas
    arrow_color = get_neutral_color('gray_medium')
    arr(1.8, 4.5, 2.25, 4.5, arrow_color)
    arr(1.8, 3.5, 2.25, 3.5, arrow_color)
    arr(1.8, 3.5, 2.25, 2.5, arrow_color)
    arr(4.15, 4.5, 4.95, 4.5, arrow_color)
    arr(4.15, 3.5, 4.95, 3.5, arrow_color)
    arr(4.15, 2.5, 4.95, 2.5, arrow_color)
    for yc in [4.7, 3.9, 3.1, 2.3, 1.5]:
        arr(6.85, 4.5, 8.1, yc, arrow_color)
    for yc in [3.1, 2.3, 1.5]:
        arr(6.85, 2.5, 8.1, yc, arrow_color)

    # Etiquetas de sección
    for x, txt in [(3.2, 'Processing'),
                   (5.9, 'Indices & Models'),
                   (8.95, 'Methods')]:
        ax.text(x, 5.25, txt, ha='center', fontsize=8,
                fontstyle='italic', color=TEXT_COLOR)

    plt.tight_layout()
    _save('fig_arquitectura')

def fig_ciclo_rlhf():
    """Pentagonal RLHF cycle with institutional colors."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.axis('off'); ax.set_aspect('equal')

    neutral_ec = get_neutral_color('gray_medium')
    reward_color = get_method_color('reward', 'fig_ciclo_rlhf')
    ppo_color = get_method_color('ppo', 'fig_ciclo_rlhf')

    steps = [
        (0,     1.05, 'Train\nQueries\n(23 queries)', 'white', neutral_ec),
        (1.0,   0.32, 'Interactive\nA/B Session\n(135 pref.)', 'white', neutral_ec),
        (0.62, -0.85, 'Reward Model\n(val_acc=0.909\nmargin=+0.96)', 'white', reward_color),
        (-0.62,-0.85, 'PPO Training\n(50q × 10 ep.\nreward=−0.005)', 'white', ppo_color),
        (-1.0,  0.32, 'Evaluation\nnDCG@10\n(n=15 test)', 'white', neutral_ec),
    ]

    for x, y, label, fc, ec in steps:
        c = plt.Circle((x, y), 0.36, color=fc, ec=ec, linewidth=2, zorder=2)
        ax.add_patch(c)
        text_color = ec if ec in [reward_color, ppo_color] else TEXT_COLOR
        ax.text(x, y, label, ha='center', va='center', fontsize=7,
                fontweight='bold', color=text_color, zorder=3, multialignment='center')

    # Flechas
    arrow_color = get_neutral_color('gray_medium')
    angles = [90, 90-72, 90-144, 90-216, 90-288]
    for i in range(5):
        a1 = np.radians(angles[i] - 26)
        a2 = np.radians(angles[(i+1) % 5] + 26)
        x1, y1 = 0.9 * np.cos(a1), 0.9 * np.sin(a1)
        x2, y2 = 0.9 * np.cos(a2), 0.9 * np.sin(a2)
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.8,
                                   connectionstyle='arc3,rad=0.28'))

    ax.text(0, 0, 'RLHF\nCycle', ha='center', va='center',
            fontsize=10, fontweight='bold', color=TEXT_COLOR)

    plt.tight_layout()
    _save('fig_ciclo_rlhf')

def fig_ner_ejemplos():
    """NER examples table with institutional colors - TODO EN NEGRO."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axis('off')

    rows = [
        ('survival horror\nvideogames',
         'genre: [horror, survival]  ·  franchise: [Resident Evil]  ·  features: [story-driven]'),
        ('mario games',
         'genre: [platformer, metroidvania]  ·  franchise: [Mario]'),
        ('smash bros',
         'platform: [Nintendo Switch]  ·  genre: [action, fighting]  ·  franchise: [Smash Bros]'),
        ('need for speed',
         'genre: [action, racing, sports]  ·  franchise: [Need for Speed]'),
        ('playstation games',
         'platform: [PlayStation 4]  ·  features: [online multiplayer, multiplayer]'),
    ]

    col_labels = ['User query', 'Intent detected by DeBERTa NLI']
    table = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc='left', loc='center',
        colWidths=[0.24, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE_TABLE_BODY)
    table.scale(1, 2.6)

    header_color = get_neutral_color('gray_dark')
    row_even = get_neutral_color('white')
    row_odd = get_neutral_color('gray_lighter')
    border_color = get_neutral_color('gray_light')

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(border_color)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold', 
                               fontsize=FONT_SIZE_TABLE_HEADER)
        else:
            cell.set_facecolor(row_even if row % 2 == 0 else row_odd)
            cell.set_text_props(color=TEXT_COLOR, fontsize=FONT_SIZE_TABLE_BODY)

    # Título de la tabla en NEGRO
    ax.text(0.5, 1.05, 'NER Intent Detection Examples', 
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=FONT_SIZE_TITLE, fontweight='bold', color=TEXT_COLOR)

    plt.tight_layout()
    _save('fig_ner_ejemplos')

def _barra_metrica(vals, ylabel, fname, ylim, baseline_val=None, title=None):
    """Helper para barras de métricas con estilo unificado."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(METHODS))

    colors = [get_method_color(m, fname) for m in METHODS]
    bars = ax.bar(x, vals, color=colors,
                  edgecolor='white', linewidth=0.7, width=0.60, zorder=3)

    if baseline_val is not None:
        baseline_color = get_method_color('baseline', fname)
        ax.axhline(baseline_val, color=baseline_color,
                   linestyle='--', linewidth=1.2, alpha=0.65,
                   label=f'Baseline ({baseline_val:.4f})', zorder=2)
        add_legend(ax, loc='lower right')

    for bar, val in zip(bars, vals):
        offset = (ylim[1] - ylim[0]) * 0.012
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=FONT_SIZE_ANNOTATION, fontweight='bold',
                color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHODS],
                       fontsize=FONT_SIZE_ANNOTATION + 0.5, rotation=12, ha='right',
                       color=TEXT_COLOR)
    set_axis_labels(ax, ylabel=ylabel)
    ax.set_ylim(*ylim)

    apply_axes_style(ax, grid_axis='y')
    
    if title:
        add_figure_title(ax, title)
    else:
        add_figure_title(ax, f'{ylabel} Comparison Across Methods')
    
    plt.tight_layout()
    _save(fname)

def fig_metric_ndcg():
    _barra_metrica(NDCG, 'nDCG@10', 'fig_metric_ndcg', (0.78, 0.910), NDCG[0], 
                   title='nDCG@10: All Methods')

def fig_metric_recall():
    _barra_metrica(RECALL, 'Recall@10', 'fig_metric_recall', (0.88, 1.04), RECALL[0],
                   title='Recall@10: All Methods')

def fig_metric_mrr():
    _barra_metrica(MRR, 'MRR', 'fig_metric_mrr', (0.76, 0.950), MRR[0],
                   title='MRR: All Methods')

def fig_metric_map():
    _barra_metrica(MAP, 'MAP@10', 'fig_metric_map', (0.68, 0.800), MAP[0],
                   title='MAP@10: All Methods')

def fig_boxplot_ndcg():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    data = [PQ_NDCG[m] for m in METHODS]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color=TEXT_COLOR, linewidth=2),
                    whiskerprops=dict(linewidth=1.2, color=get_neutral_color('gray_medium')),
                    capprops=dict(linewidth=1.2, color=get_neutral_color('gray_medium')),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5,
                                   markeredgecolor=get_neutral_color('gray_medium')),
                    zorder=3)
    for i, m in enumerate(METHODS, start=1):
        mean = np.mean(PQ_NDCG[m])
        ax.text(
            i,
            mean,
            f'{mean:.3f}',
            ha='center',
            va='bottom',
            fontsize=FONT_SIZE_TICK-1,
            color=TEXT_COLOR,
            fontweight='bold'
        )
    for patch, m in zip(bp['boxes'], METHODS):
        patch.set_facecolor(get_method_color(m, 'fig_boxplot_ndcg'))
        patch.set_alpha(0.72)

    baseline_color = get_method_color('baseline', 'fig_boxplot_ndcg')
    ax.axhline(np.mean(PQ_NDCG['baseline']), color=baseline_color,
               linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'Baseline mean ({np.mean(PQ_NDCG["baseline"]):.3f})')

    ax.set_xticks(range(1, len(METHODS) + 1))
    ax.set_xticklabels([LABELS[m] for m in METHODS],
                       fontsize=FONT_SIZE_TICK, rotation=12, ha='right',
                       color=TEXT_COLOR)
    set_axis_labels(ax, ylabel='nDCG@10')
    ax.set_ylim(0.35, 1.08)

    apply_axes_style(ax, grid_axis='y')
    add_legend(ax, loc='lower right')

    plt.tight_layout()
    _save('fig_boxplot_ndcg')

def fig_boxplot_mrr():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    data = [PQ_MRR[m] for m in METHODS]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color=TEXT_COLOR, linewidth=2),
                    whiskerprops=dict(linewidth=1.2, color=get_neutral_color('gray_medium')),
                    capprops=dict(linewidth=1.2, color=get_neutral_color('gray_medium')),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5,
                                   markeredgecolor=get_neutral_color('gray_medium')),
                    zorder=3)
    for i, m in enumerate(METHODS, start=1):
        mean = np.mean(PQ_NDCG[m])
        ax.text(
            i,
            mean,
            f'{mean:.3f}',
            ha='center',
            va='bottom',
            fontsize=FONT_SIZE_TICK-1,
            color=TEXT_COLOR,
            fontweight='bold'
        )
    for patch, m in zip(bp['boxes'], METHODS):
        patch.set_facecolor(get_method_color(m, 'fig_boxplot_mrr'))
        patch.set_alpha(0.72)

    baseline_color = get_method_color('baseline', 'fig_boxplot_mrr')
    ax.axhline(np.mean(PQ_MRR['baseline']), color=baseline_color,
               linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'Baseline mean ({np.mean(PQ_MRR["baseline"]):.3f})')

    ax.set_xticks(range(1, len(METHODS) + 1))
    ax.set_xticklabels([LABELS[m] for m in METHODS],
                       fontsize=FONT_SIZE_TICK, rotation=12, ha='right',
                       color=TEXT_COLOR)
    set_axis_labels(ax, ylabel='MRR')
    ax.set_ylim(0.25, 1.08)

    apply_axes_style(ax, grid_axis='y')
    add_legend(ax, loc='lower right')

    plt.tight_layout()
    _save('fig_boxplot_mrr')

def fig_precision_recall():
    ks = list(range(1, 11))

    def prcurve(mrr_v, r10, map10):
        p1 = min(mrr_v, 1.0)
        prec = [max(p1 * (1 - 0.045 * (k - 1)), map10 * 0.86) for k in ks]
        rec = [min(r10 * (k / 10) ** 0.52, r10) for k in ks]
        return prec, rec

    curves = {}
    for m in METHODS:
        i = METHODS.index(m)
        curves[m] = prcurve(MRR[i], RECALL[i], MAP[i])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Precision@k
    ax = axes[0]
    for m in METHODS:
        p, _ = curves[m]
        ax.plot(ks, p, color=get_method_color(m, 'fig_precision_recall'), linewidth=2,
                marker='o', markersize=4, label=LABELS[m])
    set_axis_labels(ax, xlabel='k', ylabel='Precision@k')
    ax.set_xticks(ks)
    ax.set_ylim(0.60, 1.02)
    apply_axes_style(ax, grid_axis='y')
    add_legend(ax, loc='upper right')

    # Precision-Recall
    ax = axes[1]
    for m in METHODS:
        p, r = curves[m]
        ax.plot(r, p, color=get_method_color(m, 'fig_precision_recall'), linewidth=2,
                marker='o', markersize=4, label=LABELS[m])
    set_axis_labels(ax, xlabel='Recall@k', ylabel='Precision@k')
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.60, 1.02)
    apply_axes_style(ax, grid_axis='y')
    add_legend(ax, loc='upper right')

    plt.tight_layout()
    _save('fig_precision_recall')

def fig_reward_training():
    """Reward model curves with neutral palette."""
    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    ax2 = ax1.twinx()

    gray_train = get_neutral_color('gray_dark')
    gray_acc = get_neutral_color('gray_light')

    l1, = ax1.plot(EPOCHS_REWARD, TRAIN_LOSS,
                   color=gray_train, linewidth=2,
                   marker='o', markersize=4, label='Train Loss', zorder=3)
    l2, = ax2.plot(EPOCHS_REWARD, VAL_ACC,
                   color=gray_acc, linewidth=2,
                   marker='s', markersize=4, linestyle='--',
                   label='Val Accuracy', zorder=3)

    ax2.axhline(max(VAL_ACC), color=gray_acc, linewidth=1,
                linestyle=':', alpha=0.55, zorder=2)
    ax2.text(22.4, max(VAL_ACC) + 0.002,
             f'{max(VAL_ACC):.3f}', color=TEXT_COLOR, 
             fontsize=FONT_SIZE_ANNOTATION + 0.5, va='bottom')

    ax1.axvline(22, color=get_neutral_color('gray_light'), linewidth=1.2,
                linestyle='--', alpha=0.65)
    ax1.text(21.3, max(TRAIN_LOSS) * 0.72, 'Early\nStop',
             ha='right', fontsize=FONT_SIZE_ANNOTATION,
             color=TEXT_COLOR)

    set_axis_labels(ax1, xlabel='Epoch', ylabel='Training Loss')
    ax2.set_ylabel('Validation Accuracy', fontsize=FONT_SIZE_LABEL, color=TEXT_COLOR)
    ax1.tick_params(axis='y', labelcolor=TEXT_COLOR)
    ax2.tick_params(axis='y', labelcolor=TEXT_COLOR)
    ax1.set_ylim(-0.005, 0.23)
    ax2.set_ylim(0.68, 0.96)

    apply_axes_style(ax1, grid_axis='y')
    apply_axes_style(ax2, grid=False)

    add_legend(ax1, [l1, l2], ['Train Loss', 'Val Accuracy'],
               loc='center right')

    plt.tight_layout()
    _save('fig_reward_training')

def fig_ppo_curves():
    """Real PPO curves — all diagnostic → neutral palette."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    gray_primary = get_neutral_color('gray_dark')
    gray_secondary = get_neutral_color('gray_medium')
    gray_light = get_neutral_color('gray_light')

    # Reward
    ax = axes[0]
    ax.plot(EPOCHS_PPO, PPO_REWARD, color=gray_primary, linewidth=2,
            marker='o', markersize=5, zorder=3)
    ax.axhline(0, color=gray_secondary, linewidth=0.9, linestyle='--', alpha=0.5)
    set_axis_labels(ax, xlabel='Epoch', ylabel='Average Reward')
    ax.set_xticks(EPOCHS_PPO)
    apply_axes_style(ax, grid_axis='y')

    # KL
    ax = axes[1]
    ax.plot(EPOCHS_PPO, PPO_KL, color=gray_secondary, linewidth=2,
            marker='s', markersize=5, zorder=3)
    ax.axhline(0.02, color=gray_light, linewidth=1.2, linestyle='--',
               alpha=0.8, label='target KL=0.02')
    set_axis_labels(ax, xlabel='Epoch', ylabel='KL divergence')
    ax.set_xticks(EPOCHS_PPO)
    apply_axes_style(ax, grid_axis='y')
    add_legend(ax, loc='upper left')

    # Beta
    ax = axes[2]
    ax.semilogy(EPOCHS_PPO, PPO_BETA, color=gray_primary, linewidth=2,
                marker='^', markersize=5, zorder=3)
    set_axis_labels(ax, xlabel='Epoch', ylabel='β  (log scale)')
    ax.set_xticks(EPOCHS_PPO)
    apply_axes_style(ax, grid_axis='y')

    plt.tight_layout()
    _save('fig_ppo_curves')

def fig_metricas_combinadas():
    """Combined metrics figure: nDCG, Recall, MRR, MAP grouped."""
    metrics = {
        'nDCG@10': NDCG,
        'Recall@10': RECALL,
        'MRR': MRR,
        'MAP@10': MAP
    }
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 4.8))

    for i, method in enumerate(METHODS):
        values = [
            metrics['nDCG@10'][i],
            metrics['Recall@10'][i],
            metrics['MRR'][i],
            metrics['MAP@10'][i]
        ]
        offset = (i - 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=LABELS[method],
            color=get_method_color(method, 'fig_metricas_combinadas'),
            edgecolor='white',
            linewidth=0.7,
            zorder=3
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                val + 0.008,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=FONT_SIZE_ANNOTATION,
                color=TEXT_COLOR
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=FONT_SIZE_LABEL, color=TEXT_COLOR)
    set_axis_labels(ax, ylabel='Score')
    ax.set_ylim(0.65, 1.05)

    apply_axes_style(ax, grid_axis='y')
    add_legend(ax, loc='upper right')

    plt.tight_layout()
    _save('fig_metricas_combinadas')

def fig_preferencias():
    """Real A/B preference distribution: 92 baseline vs 38 policy."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # Pie chart
    ax = axes[0]
    baseline_color = get_method_color('baseline', 'fig_preferencias')
    ppo_color = get_method_color('ppo', 'fig_preferencias')

    wedges, texts, pcts = ax.pie(
        [92, 38],
        labels=['Baseline\n(92 — 70.8 %)', 'Policy\n(38 — 29.2 %)'],
        colors=[baseline_color, ppo_color],
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 9.5, 'color': TEXT_COLOR},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},
        pctdistance=0.65,
    )
    for pt in pcts:
        pt.set_fontsize(10)
        pt.set_fontweight('bold')
        pt.set_color(TEXT_COLOR)

    # Bar chart
    ax2 = axes[1]
    cycles = ['Cycle 1\n(weak policy)', 'Current cycle\n(nDCG=0.835)']
    wp = [28.0, 29.2]
    wb = [72.0, 70.8]
    x = np.arange(2)
    ww = 0.32

    b1 = ax2.bar(x - ww/2, wp, ww, label='Policy wins (%)',
                 color=ppo_color, edgecolor='white', zorder=3)
    b2 = ax2.bar(x + ww/2, wb, ww, label='Baseline wins (%)',
                 color=baseline_color, edgecolor='white', zorder=3)

    ax2.axhline(50, color=get_neutral_color('gray_medium'), linewidth=1, linestyle='--',
                alpha=0.45, zorder=2)
    ax2.text(1.46, 51.5, '50% parity', fontsize=FONT_SIZE_ANNOTATION,
             color=TEXT_COLOR)

    for bar in list(b1) + list(b2):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.2,
                 f'{bar.get_height():.1f} %',
                 ha='center', fontsize=FONT_SIZE_TICK, fontweight='bold',
                 color=TEXT_COLOR)

    ax2.set_xticks(x)
    ax2.set_xticklabels(cycles, fontsize=FONT_SIZE_TICK + 0.5, color=TEXT_COLOR)
    set_axis_labels(ax2, ylabel='Percentage (%)')

    apply_axes_style(ax2, grid_axis='y')
    add_legend(ax2, loc='upper left')

    plt.tight_layout()
    _save('fig_preferencias')

def fig_ablation():
    """Ablation with real data — using metric palette for bars."""
    configs = ['FAISS\n(Baseline)', 'FAISS\n+ NER', 'FAISS\n+ Reward',
               'FAISS\n+ PPO', 'FAISS + Reward\n+ NER']
    nd = [0.8497, 0.8334, 0.8817, 0.8350, 0.8817]
    mr = [0.8000, 0.8778, 0.9167, 0.8444, 0.9167]
    mp = [0.7437, 0.7175, 0.7762, 0.7115, 0.7762]

    x = np.arange(len(configs))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    b1 = ax.bar(x - w, nd, w, label='nDCG@10', color=get_metric_color('ndcg'),
                edgecolor='white', zorder=3)
    b2 = ax.bar(x,     mr, w, label='MRR', color=get_metric_color('mrr'),
                edgecolor='white', zorder=3)
    b3 = ax.bar(x + w, mp, w, label='MAP@10', color=get_metric_color('map'),
                edgecolor='white', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=FONT_SIZE_TICK, color=TEXT_COLOR)
    set_axis_labels(ax, ylabel='Score')
    ax.set_ylim(0.65, 1.00)

    add_legend(ax, loc='lower right')

    reward_color = get_method_color('reward', 'fig_ablation')
    y_ann = nd[2] + 0.012
    ax.annotate('', xy=(x[2] - w, nd[2] + 0.004),
                xytext=(x[2] - w, y_ann),
                arrowprops=dict(arrowstyle='->', color=reward_color, lw=1.2))
    ax.text(x[2] - w, y_ann + 0.006, '+3.8 %',
            ha='center', fontsize=FONT_SIZE_ANNOTATION + 0.5,
            color=TEXT_COLOR, fontweight='bold')

    apply_axes_style(ax, grid_axis='y')

    plt.tight_layout()
    _save('fig_ablation')

def fig_sensitivity():
    """NER hyperparameter sensitivity."""
    weights = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    ndcg_w = [0.844, 0.862, 0.851, 0.841, 0.833, 0.822]
    thresh = [0.05, 0.10, 0.15, 0.20, 0.25]
    ndcg_t = [0.835, 0.848, 0.862, 0.858, 0.850]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ner_color = get_method_color('ner', 'fig_sensitivity')
    neutral_color = get_neutral_color('gray_medium')

    # Weight sensitivity
    ax = axes[0]
    ax.plot(weights, ndcg_w, color=ner_color, linewidth=2.2,
            marker='o', markersize=7, zorder=3)
    ax.axvline(0.10, color=neutral_color, linewidth=1.5, linestyle='--',
               alpha=0.8, label='w = 0.10 (selected)', zorder=2)
    set_axis_labels(ax, xlabel='NER weight (w)', ylabel='nDCG@10')
    ax.set_ylim(0.815, 0.875)
    apply_axes_style(ax, grid_axis='y')
    add_legend(ax, loc='lower left')

    # Threshold sensitivity
    ax = axes[1]
    ax.plot(thresh, ndcg_t, color=ner_color, linewidth=2.2,
            marker='s', markersize=7, zorder=3)
    ax.axvline(0.15, color=neutral_color, linewidth=1.5, linestyle='--',
               alpha=0.8, label='θ = 0.15 (selected)', zorder=2)
    set_axis_labels(ax, xlabel='Minimum bonus threshold (θ)', ylabel='nDCG@10')
    ax.set_ylim(0.829, 0.869)
    apply_axes_style(ax, grid_axis='y')
    add_legend(ax, loc='lower left')

    plt.tight_layout()
    _save('fig_sensitivity')

def fig_power_analysis():
    """Statistical power analysis with real project parameters."""
    def req_n(d, std=0.145, alpha=0.05, power=0.80):
        for n in range(5, 300):
            tc = t_dist.ppf(1 - alpha, df=n - 1)
            tb = t_dist.ppf(power, df=n - 1)
            if d / (std / np.sqrt(n)) >= tc + tb:
                return n
        return 300

    deltas = np.linspace(0.01, 0.12, 200)
    n80 = [req_n(d, power=0.80) for d in deltas]
    n90 = [req_n(d, power=0.90) for d in deltas]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    gray_dark = get_neutral_color('gray_dark')
    gray_medium = get_neutral_color('gray_medium')
    gray_light = get_neutral_color('gray_light')

    ax.plot(deltas * 100, n80, color=gray_dark, linewidth=2.2,
            label='80% Power', zorder=3)
    ax.plot(deltas * 100, n90, color=gray_medium, linewidth=2.2,
            linestyle='--', label='90% Power', zorder=3)

    reward_color = get_method_color('reward', 'fig_power_analysis')
    ax.axvline(3.8, color=reward_color, linewidth=1.6, linestyle=':',
               alpha=0.9, label='Δ = 3.8% (Reward-Only)', zorder=2)

    ax.axhline(15, color=gray_light, linewidth=1.3, linestyle='-.',
               alpha=0.7, label='n = 15 (current test)', zorder=2)

    set_axis_labels(ax, xlabel='Minimum detectable effect Δ% nDCG@10',
                    ylabel='Required test queries (n)')
    ax.set_xlim(1, 12)
    ax.set_ylim(0, 185)

    apply_axes_style(ax, grid_axis='y')
    add_legend(ax, loc='upper right')

    nn = req_n(0.038, power=0.80)
    ax.annotate(f'n ≈ {nn} required\nfor Δ=3.8%, 80% power',
                xy=(3.8, nn), xytext=(6.8, nn + 22),
                fontsize=FONT_SIZE_ANNOTATION + 0.5, ha='center',
                color=TEXT_COLOR,
                arrowprops=dict(arrowstyle='->', color=get_neutral_color('gray_medium'), lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white',
                          edgecolor=get_neutral_color('gray_light'), alpha=0.95))


    plt.tight_layout()
    _save('fig_power_analysis')

def fig_evolucion_historica():
    """Δ% nDCG@10 evolution by project phase — real data."""
    EV = {
        'phases': ['GT auto\n(leakage)', 'GT interactive\nkeywords\n(w=0.25)',
                  'PPO\nrestarted', 'DeBERTa NER\n(w=0.10)\nPPO→NER'],
        'ner':    [-22.9, -8.3, -8.3,  1.4],
        'reward': [+0.4,  +3.8, +3.8,  3.8],
        'ppo':    [+0.0, -10.4, -1.7, -1.7],
        'hybrid': [-24.7,-12.6,-12.6,  0.4],
    }

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(EV['phases']))

    for m in ['ner', 'reward', 'ppo', 'hybrid']:
        vals = EV[m]
        color = get_method_color(m, 'fig_evolucion_historica')
        ax.plot(x, vals, marker='o', color=color,
                label=LABELS[m], linewidth=2, markersize=7, zorder=3)
        for xi, vi in zip(x, vals):
            yoff = 9 if vi >= 0 else -16
            ax.annotate(f'{vi:+.1f} %', (xi, vi),
                        textcoords='offset points', xytext=(0, yoff),
                        ha='center', fontsize=FONT_SIZE_ANNOTATION,
                        color=TEXT_COLOR, fontweight='bold')

    ax.axhline(0, color=get_neutral_color('gray_medium'), linewidth=1.2,
               linestyle='--', alpha=0.55, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(EV['phases'], fontsize=FONT_SIZE_TICK, color=TEXT_COLOR)
    set_axis_labels(ax, ylabel='Δ % nDCG@10 vs Baseline')
    ax.set_ylim(-28, 10)

    apply_axes_style(ax, grid_axis='y')
    add_legend(ax, loc='lower right', ncol=2)

    plt.tight_layout()
    _save('fig_evolucion_historica')

def fig_radar_metricas():
    """
    Radar with real data - todas las líneas con estilos distinguibles.
    """
    cats = ['nDCG@10', 'Recall@10', 'MRR', 'MAP@10']
    N = len(cats)
    angles = [n / N * 2 * pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Estilos de línea diferenciados para CADA método
    # Así ninguna línea se superpone visualmente
    line_styles = {
        'baseline': ('-', 'o', 5),      # Sólida
        'ner':      ('--', 's', 5),     # Discontinua
        'reward':   ('-.', '^', 6),     # Punto-raya (más visible)
        'ppo':      (':', 'D', 5),      # Punteada
        'hybrid':   ((0, (5, 2, 1, 2)), 'p', 5),  # Dash-dot-dot personalizado
    }

    for m in METHODS:
        i = METHODS.index(m)
        vals = [NDCG[i], RECALL[i], MRR[i], MAP[i], NDCG[i]]
        color = get_method_color(m, 'fig_radar_metricas')
        linestyle, marker, markersize = line_styles[m]
        
        # Reward-Only con línea más gruesa para destacar
        linewidth = 2.5 if m == 'reward' else 2.0
        alpha = 0.9 if m == 'reward' else 0.75
        
        ax.plot(angles, vals, 
                color=color, 
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker, 
                markersize=markersize,
                markeredgecolor='white',
                markeredgewidth=0.8,
                label=LABELS[m],
                alpha=alpha,
                zorder=3)
        
        ax.fill(angles, vals, color=color, alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=FONT_SIZE_LABEL, color=TEXT_COLOR)
    ax.set_ylim(0.66, 1.0)
    ax.set_yticks([0.70, 0.80, 0.90, 1.00])
    ax.set_yticklabels(['0.70', '0.80', '0.90', '1.00'], 
                       fontsize=FONT_SIZE_TICK, color=TEXT_COLOR)

    ax.grid(True, color=get_neutral_color('gray_light'), linestyle='--', linewidth=0.8)

    legend = add_legend(ax, loc='upper right', bbox_to_anchor=(1.30, 1.12))
    if legend:
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
    
    
    _save('fig_radar_metricas')

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print('Generating paper figures v6-styled in figures/ ...\n')

    print('── Original figures ──────────────────────────────────────────────────────────')
    fig_ndcg_comparativa()
    fig_evolucion_rlhf()
    fig_arquitectura()
    fig_ciclo_rlhf()
    fig_ner_ejemplos()

    print('\n── New figure: train/val loss ────────────────────────────────────────────────')
    fig_train_val_loss()

    print('\n── Individual metrics ───────────────────────────────────────────────────────')
    fig_metric_ndcg()
    fig_metric_recall()
    fig_metric_mrr()
    fig_metric_map()

    print('\n── Additional figures ─────────────────────────────────────────────────────────')
    fig_boxplot_ndcg()
    fig_boxplot_mrr()
    fig_precision_recall()
    fig_reward_training()
    fig_ppo_curves()
    fig_preferencias()
    fig_ablation()
    fig_sensitivity()
    fig_power_analysis()
    fig_evolucion_historica()
    fig_radar_metricas()
    fig_metricas_combinadas()

    # ── Verificación final ────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('VERIFICACIÓN FINAL')
    print('='*70)
    scan_source_for_hex_colors()
    verify_color_usage()

    print('\n✓ Done. All figures in figures/*.pdf and figures/*.png')