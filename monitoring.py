import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict

class RLHFMonitor:
    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = Path(feedback_dir)
    
    def load_feedback_data(self):
        """Carga todos los feedbacks disponibles"""
        feedback_files = self.feedback_dir.glob("*.jsonl")
        samples = []
        for file in feedback_files:
            with open(file, 'r', encoding='utf-8') as f:
                samples.extend([json.loads(line) for line in f])
        return samples
    
    def generate_stats_report(self):
        """Genera reporte completo de estadísticas"""
        feedbacks = self.load_feedback_data()
        if not feedbacks:
            return {"error": "No feedback data available"}
        
        # Estadísticas básicas
        stats = {
            "total_feedbacks": len(feedbacks),
            "by_language": defaultdict(int),
            "avg_rating": 0,
            "rating_distribution": defaultdict(int),
            "domain_term_usage": defaultdict(int)
        }
        
        # Calcular métricas
        total_rating = 0
        for fb in feedbacks:
            stats['by_language'][fb['lang']] += 1
            stats['rating_distribution'][fb['rating']] += 1
            total_rating += fb['rating']
            
            for term in fb.get('domain_terms', {}):
                stats['domain_term_usage'][term] += 1
        
        stats['avg_rating'] = total_rating / len(feedbacks)
        
        return stats
    
    def plot_metrics_trend(self, output_file: str = "monitoring/metrics_trend.png"):
        """Genera gráfico de tendencia de métricas clave"""
        feedbacks = self.load_feedback_data()
        if not feedbacks:
            return
        
        # Agrupar por fecha
        daily_data = defaultdict(list)
        for fb in feedbacks:
            date = datetime.fromisoformat(fb['timestamp']).strftime('%Y-%m-%d')
            daily_data[date].append(fb['rating'])
        
        # Preparar datos para plotting
        dates = sorted(daily_data.keys())
        avg_ratings = [np.mean(daily_data[date]) for date in dates]
        
        # Crear gráfico
        plt.figure(figsize=(10, 5))
        plt.plot(dates, avg_ratings, marker='o')
        plt.title('RLHF Feedback Trend - Average Rating')
        plt.xlabel('Date')
        plt.ylabel('Average Rating (1-5)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        
        # Guardar gráfico
        Path(output_file).parent.mkdir(exist_ok=True)
        plt.savefig(output_file)
        plt.close()