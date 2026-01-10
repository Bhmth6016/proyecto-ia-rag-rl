# src/web_interface/app.py
"""
Interfaz Web para Sistema RAG+RL - Permite queries y feedback para RLHF
"""
from flask import Flask, render_template, request, jsonify, session
import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# Configurar paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))

app = Flask(__name__)
app.secret_key = 'rag_rl_secret_key_2024'

# Importar componentes del sistema
try:
    from data.canonicalizer import ProductCanonicalizer
    from data.vector_store import VectorStore
    from src.query.understanding import QueryUnderstanding
    from ranking.ranking_engine import StaticRankingEngine
    from src.ranking.rl_ranker import RLHFAgent, LinUCB
    from consistency_checker import ConsistencyChecker
    import yaml
except ImportError as e:
    print(f"Error de importación: {e}")
    # Crear stubs para desarrollo
    class MockComponent:
        pass
    ProductCanonicalizer = MockComponent
    VectorStore = MockComponent
    QueryUnderstanding = MockComponent
    StaticRankingEngine = MockComponent
    RLHFAgent = MockComponent
    LinUCB = MockComponent


class InteractiveRAGRLSystem:
    """Sistema interactivo para interfaz web"""
    
    def __init__(self, config_path: str = "config/paper_experiment.yaml"):
        self.config = self._load_config(config_path)
        self.user_sessions = {}  # user_id -> session_data
        self.interaction_log = []
        
        # Inicializar componentes
        self._initialize_components()
        
        # Estadísticas
        self.stats = {
            'total_queries': 0,
            'total_feedback': 0,
            'users': set(),
            'avg_rating': 0.0
        }
    
    def _load_config(self, config_path: str):
        """Carga configuración"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            # Configuración por defecto
            return {
                'experiment': {'name': 'Interactive RAG+RL'},
                'embedding': {'model': 'all-MiniLM-L6-v2', 'dimension': 384},
                'ranking': {
                    'baseline_weights': {
                        'content_similarity': 0.4,
                        'title_similarity': 0.2,
                        'category_exact_match': 0.15,
                        'rating_normalized': 0.1,
                        'price_available': 0.05
                    }
                },
                'rlhf': {'alpha': 0.1}
            }
    
    def _initialize_components(self):
        """Inicializa componentes del sistema"""
        print("🔄 Inicializando componentes del sistema interactivo...")
        
        try:
            # Canonicalizer y Vector Store
            self.canonicalizer = ProductCanonicalizer(
                embedding_model=self.config['embedding']['model']
            )
            
            # Cargar datos REALES
            self.products = self._load_real_products()
            
            # Crear productos canónicos
            self.canonical_products = self.canonicalizer.batch_canonicalize(
                self.products[:100000]  # Limitar para demo interactiva
            )
            
            # Vector Store
            self.vector_store = VectorStore(
                dimension=self.config['embedding']['dimension']
            )
            self.vector_store.build_index(self.canonical_products)
            
            # Query Understanding
            self.query_understanding = QueryUnderstanding()
            
            # Ranking Engine
            weights = self.config['ranking']['baseline_weights']
            self.ranking_engine = StaticRankingEngine(weights=weights)
            
            # RLHF Agent
            self.rlhf_agent = self._create_rlhf_agent()
            
            print(f"✅ Sistema interactivo inicializado:")
            print(f"   • Productos: {len(self.canonical_products)}")
            print(f"   • Dimensiones: {self.config['embedding']['dimension']}")
            print(f"   • RLHF alpha: {self.config['rlhf'].get('alpha', 0.1)}")
            
        except Exception as e:
            print(f"⚠️  Error inicializando componentes: {e}")
            print("   Creando componentes simulados para demo...")
            self._create_mock_components()
    
    def _load_real_products(self):
        """Carga productos REALES de los archivos .jsonl"""
        print("📥 Cargando productos reales...")
        
        products = []
        raw_dir = Path("data/raw")
        jsonl_files = list(raw_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            print("⚠️  No hay archivos .jsonl, creando datos de ejemplo")
            return self._create_sample_products()
        
        # Procesar primeros 2 archivos
        for file_path in jsonl_files[:2]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 250:  # 250 productos por archivo
                            break
                        
                        try:
                            data = json.loads(line.strip())
                            product = {
                                'id': data.get('asin') or f"prod_{i:06d}",
                                'title': data.get('title', '')[:100],
                                'description': data.get('description', '')[:200],
                                'price': self._extract_price(data),
                                'category': self._extract_category(data),
                                'brand': data.get('brand', ''),
                                'rating': self._extract_rating(data),
                                'rating_count': data.get('rating_count')
                            }
                            
                            if product['title']:
                                products.append(product)
                                
                        except json.JSONDecodeError:
                            continue
                
                print(f"   ✓ {file_path.name}: {min(250, i)} productos")
                
            except Exception as e:
                print(f"   ✗ Error en {file_path.name}: {e}")
        
        print(f"✅ Total productos cargados: {len(products)}")
        return products
    
    def _extract_price(self, data):
        """Extrae precio"""
        price_keys = ['price', 'Price', 'list_price']
        for key in price_keys:
            if key in data and data[key]:
                try:
                    price_str = str(data[key])
                    import re
                    match = re.search(r'(\d+\.?\d*)', price_str.replace(',', ''))
                    if match:
                        return float(match.group(1))
                except:
                    continue
        return None
    
    def _extract_category(self, data):
        """Extrae categoría"""
        category_keys = ['main_category', 'category']
        for key in category_keys:
            if key in data and data[key]:
                cat = data[key]
                if isinstance(cat, str):
                    return cat.split('|')[0].strip() if '|' in cat else cat.strip()
        return "General"
    
    def _extract_rating(self, data):
        """Extrae rating"""
        rating_keys = ['rating', 'average_rating']
        for key in rating_keys:
            if key in data and data[key]:
                try:
                    rating = float(data[key])
                    return max(0.0, min(5.0, rating))
                except:
                    continue
        return None
    
    def _create_sample_products(self):
        """Crea productos de ejemplo si no hay datos reales"""
        products = []
        for i in range(100):
            products.append({
                'id': f"sample_{i:03d}",
                'title': f"Producto de Ejemplo {i+1}",
                'description': f"Descripción del producto de ejemplo {i+1}",
                'price': 10.0 + i * 5,
                'category': ['Electronics', 'Books', 'Home'][i % 3],
                'brand': f"Brand{(i % 5) + 1}",
                'rating': 3.5 + (i % 15) * 0.1,
                'rating_count': i * 10
            })
        return products
    
    def _create_rlhf_agent(self):
        """Crea agente RLHF para aprendizaje interactivo"""
        
        def feature_extractor(query_features, product):
            """Extrae características para RLHF"""
            features = {}
            
            # Características del producto
            if hasattr(product, 'price'):
                features['price_available'] = 1.0 if product.price else 0.0
            
            if hasattr(product, 'rating'):
                features['has_rating'] = 1.0 if product.rating else 0.0
                if product.rating:
                    features['rating_value'] = product.rating / 5.0
            
            # Match con query
            if 'category' in query_features and hasattr(product, 'category'):
                features['category_match'] = 1.0 if query_features['category'] == product.category else 0.0
            
            # Similitud (si está disponible)
            if 'query_embedding' in query_features and hasattr(product, 'content_embedding'):
                sim = np.dot(query_features['query_embedding'], product.content_embedding)
                features['content_similarity'] = float(sim)
            
            return features
        
        return RLHFAgent(
            feature_extractor=feature_extractor,
            alpha=self.config['rlhf'].get('alpha', 0.1)
        )
    
    def _create_mock_components(self):
        """Crea componentes simulados para demo"""
        class MockProduct:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
                self.title_embedding = np.random.randn(384)
                self.content_embedding = np.random.randn(384)
        
        # Productos simulados
        self.products = self._create_sample_products()
        self.canonical_products = [MockProduct(p) for p in self.products[:100]]
        
        # Componentes mock
        self.canonicalizer = type('MockCanonicalizer', (), {
            'embedding_model': type('MockModel', (), {
                'encode': lambda self, text, **kwargs: np.random.randn(384)
            })()
        })()
        
        self.query_understanding = type('MockQueryUnderstanding', (), {
            'analyze': lambda self, query: {
                'category': 'General',
                'intent': 'search',
                'entities': [],
                'keywords': query.split()
            }
        })()
        
        self.ranking_engine = type('MockRankingEngine', (), {
            'rank_products': lambda self, **kwargs: self.canonical_products[:10]
        })()
        
        self.rlhf_agent = self._create_rlhf_agent()
    
    def process_query(self, query_text, user_id="anonymous", use_rlhf=True):
        """Procesa una query del usuario"""
        self.stats['total_queries'] += 1
        self.stats['users'].add(user_id)
        
        # 1. Análisis de la query
        query_analysis = self.query_understanding.analyze(query_text)
        
        # 2. Generar embedding de la query
        query_embedding = self.canonicalizer.embedding_model.encode(
            query_text, normalize_embeddings=True
        )
        
        # 3. Recuperación
        retrieved = self.vector_store.search(query_embedding, k=50)
        
        if not retrieved:
            return {
                'success': False,
                'error': 'No se encontraron productos',
                'query_analysis': query_analysis
            }
        
        # 4. Ranking
        if use_rlhf and self.rlhf_agent and user_id in self.user_sessions:
            # Usar RLHF si el usuario tiene sesión de aprendizaje
            user_session = self.user_sessions[user_id]
            
            # Extraer características de la query para RLHF
            query_features = {
                'category': query_analysis.get('category'),
                'intent': query_analysis.get('intent'),
                'query_embedding': query_embedding
            }
            
            # Obtener ranking con RLHF
            baseline_indices = list(range(len(retrieved)))
            rlhf_ranking = self.rlhf_agent.select_ranking(
                query_features=query_features,
                products=retrieved,
                baseline_ranking=baseline_indices
            )
            
            # Ordenar productos según ranking RLHF
            ranked_products = [retrieved[idx] for idx in rlhf_ranking[:10]]
            ranking_method = "RLHF (personalizado)"
            
        else:
            # Ranking tradicional
            ranked_products = self.ranking_engine.rank_products(
                query_embedding=query_embedding,
                query_category=query_analysis.get('category', 'General'),
                products=retrieved,
                top_k=10
            )
            ranking_method = "Baseline"
        
        # 5. Formatear resultados para la interfaz
        formatted_results = []
        for i, product in enumerate(ranked_products[:10]):
            formatted_results.append({
                'position': i + 1,
                'id': getattr(product, 'id', 'N/A'),
                'title': getattr(product, 'title', 'Sin título'),
                'description': getattr(product, 'description', '')[:100] + '...',
                'price': f"${getattr(product, 'price', 0):.2f}" if hasattr(product, 'price') and product.price else "N/A",
                'category': getattr(product, 'category', 'General'),
                'rating': f"⭐{getattr(product, 'rating', 0):.1f}" if hasattr(product, 'rating') and product.rating else "Sin rating",
                'brand': getattr(product, 'brand', ''),
                'features': getattr(product, 'features_dict', {})
            })
        
        # 6. Guardar interacción para RLHF
        interaction_id = len(self.interaction_log)
        interaction = {
            'id': interaction_id,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'query': query_text,
            'query_analysis': query_analysis,
            'results_count': len(ranked_products),
            'ranking_method': ranking_method,
            'shown_products': [p['id'] for p in formatted_results[:5]],  # IDs de productos mostrados
            'feedback_received': False
        }
        
        self.interaction_log.append(interaction)
        
        # 7. Preparar respuesta
        response = {
            'success': True,
            'query': query_text,
            'query_analysis': query_analysis,
            'results': formatted_results,
            'stats': {
                'total_found': len(retrieved),
                'shown': len(formatted_results),
                'ranking_method': ranking_method
            },
            'interaction_id': interaction_id,
            'user_id': user_id
        }
        
        return response
    
    def process_feedback(self, interaction_id, product_id, rating, user_id="anonymous"):
        """Procesa feedback del usuario para RLHF"""
        self.stats['total_feedback'] += 1
        
        # Actualizar promedio de ratings
        old_avg = self.stats['avg_rating']
        n = self.stats['total_feedback']
        self.stats['avg_rating'] = (old_avg * (n-1) + rating) / n if n > 0 else rating
        
        # Buscar la interacción
        if interaction_id >= len(self.interaction_log):
            return {'success': False, 'error': 'Interacción no encontrada'}
        
        interaction = self.interaction_log[interaction_id]
        
        # Verificar que el usuario coincida
        if interaction['user_id'] != user_id:
            return {'success': False, 'error': 'Usuario no autorizado'}
        
        # Buscar el producto en los resultados mostrados
        if product_id not in interaction['shown_products']:
            return {'success': False, 'error': 'Producto no mostrado en esta interacción'}
        
        # Obtener embedding de la query original
        query_text = interaction['query']
        query_embedding = self.canonicalizer.embedding_model.encode(
            query_text, normalize_embeddings=True
        )
        
        # Buscar productos recuperados originalmente
        retrieved = self.vector_store.search(query_embedding, k=50)
        
        # Encontrar el índice del producto seleccionado
        selected_idx = -1
        for i, product in enumerate(retrieved):
            if hasattr(product, 'id') and product.id == product_id:
                selected_idx = i
                break
        
        if selected_idx == -1:
            return {'success': False, 'error': 'Producto no encontrado en resultados originales'}
        
        # Preparar características de la query para RLHF
        query_features = {
            'category': interaction['query_analysis'].get('category'),
            'intent': interaction['query_analysis'].get('intent'),
            'query_embedding': query_embedding
        }
        
        # Actualizar RLHF agent con el feedback
        if self.rlhf_agent:
            # Inicializar sesión de usuario si no existe
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = {
                    'queries_count': 0,
                    'feedbacks_count': 0,
                    'preferences': {}
                }
            
            user_session = self.user_sessions[user_id]
            user_session['queries_count'] = user_session.get('queries_count', 0) + 1
            user_session['feedbacks_count'] = user_session.get('feedbacks_count', 0) + 1
            
            # Guardar preferencia
            if 'preferences' not in user_session:
                user_session['preferences'] = {}
            
            # Actualizar RLHF
            shown_indices = list(range(min(10, len(retrieved))))
            self.rlhf_agent.update_with_feedback(
                query_features=query_features,
                products=retrieved,
                shown_indices=shown_indices,
                selected_idx=selected_idx,
                rating=rating
            )
            
            # Guardar preferencia del usuario
            category = interaction['query_analysis'].get('category', 'General')
            if category not in user_session['preferences']:
                user_session['preferences'][category] = []
            
            user_session['preferences'][category].append({
                'product_id': product_id,
                'rating': rating,
                'timestamp': datetime.now().isoformat()
            })
        
        # Actualizar interacción
        interaction['feedback_received'] = True
        interaction['feedback'] = {
            'product_id': product_id,
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar feedback en archivo para análisis
        self._save_feedback_to_file(interaction_id, product_id, rating, user_id)
        
        return {
            'success': True,
            'message': f'Feedback recibido: Rating {rating} para producto {product_id}',
            'rlhf_updated': self.rlhf_agent is not None,
            'user_session': self.user_sessions.get(user_id, {})
        }
    
    def _save_feedback_to_file(self, interaction_id, product_id, rating, user_id):
        """Guarda feedback en archivo para análisis posterior"""
        feedback_dir = Path("data/feedback")
        feedback_dir.mkdir(exist_ok=True)
        
        feedback_file = feedback_dir / f"feedback_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        feedback_data = {
            'timestamp': datetime.now().isoformat(),
            'interaction_id': interaction_id,
            'product_id': product_id,
            'rating': rating,
            'user_id': user_id,
            'rlhf_alpha': self.config['rlhf'].get('alpha', 0.1)
        }
        
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_data) + '\n')
    
    def get_system_stats(self):
        """Obtiene estadísticas del sistema"""
        return {
            'total_queries': self.stats['total_queries'],
            'total_feedback': self.stats['total_feedback'],
            'unique_users': len(self.stats['users']),
            'average_rating': self.stats['avg_rating'],
            'total_products': len(self.canonical_products),
            'rlhf_initialized': self.rlhf_agent is not None,
            'user_sessions': len(self.user_sessions)
        }
    
    def get_user_session(self, user_id):
        """Obtiene sesión de un usuario"""
        return self.user_sessions.get(user_id, {})
    
    def get_rlhf_learning_curve(self, user_id=None):
        """Obtiene curva de aprendizaje RLHF"""
        if not self.rlhf_agent:
            return []
        
        return self.rlhf_agent.get_learning_curve()


# Inicializar sistema global
rag_rl_system = InteractiveRAGRLSystem()


# Rutas de la API
@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def api_query():
    """API para procesar queries"""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({'success': False, 'error': 'Query requerida'})
    
    query_text = data['query']
    user_id = data.get('user_id', 'anonymous')
    use_rlhf = data.get('use_rlhf', True)
    
    try:
        result = rag_rl_system.process_query(query_text, user_id, use_rlhf)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """API para recibir feedback"""
    data = request.json
    
    required_fields = ['interaction_id', 'product_id', 'rating']
    for field in required_fields:
        if field not in data:
            return jsonify({'success': False, 'error': f'Campo {field} requerido'})
    
    interaction_id = data['interaction_id']
    product_id = data['product_id']
    rating = float(data['rating'])
    user_id = data.get('user_id', 'anonymous')
    
    # Validar rating
    if rating < 1 or rating > 5:
        return jsonify({'success': False, 'error': 'Rating debe estar entre 1 y 5'})
    
    try:
        result = rag_rl_system.process_feedback(interaction_id, product_id, rating, user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """API para obtener estadísticas del sistema"""
    stats = rag_rl_system.get_system_stats()
    return jsonify({'success': True, 'stats': stats})


@app.route('/api/user/<user_id>', methods=['GET'])
def api_user(user_id):
    """API para obtener información de usuario"""
    session_data = rag_rl_system.get_user_session(user_id)
    learning_curve = rag_rl_system.get_rlhf_learning_curve(user_id)
    
    return jsonify({
        'success': True,
        'user_id': user_id,
        'session': session_data,
        'learning_curve': learning_curve
    })


@app.route('/api/debug/products', methods=['GET'])
def api_debug_products():
    """API de debug para ver productos (solo desarrollo)"""
    products_info = []
    for i, product in enumerate(rag_rl_system.canonical_products[:20]):
        products_info.append({
            'id': getattr(product, 'id', f'product_{i}'),
            'title': getattr(product, 'title', 'Sin título'),
            'category': getattr(product, 'category', 'General'),
            'price': getattr(product, 'price', None),
            'rating': getattr(product, 'rating', None)
        })
    
    return jsonify({
        'success': True,
        'total_products': len(rag_rl_system.canonical_products),
        'sample': products_info
    })


if __name__ == '__main__':
    print("🚀 Iniciando servidor web RAG+RL...")
    print("📊 Accede en: http://localhost:5000")
    print("⚙️  Sistema inicializado con RLHF para aprendizaje en tiempo real")
    
    # Mostrar estadísticas iniciales
    stats = rag_rl_system.get_system_stats()
    print(f"\n📈 Estadísticas iniciales:")
    print(f"   • Productos cargados: {stats['total_products']}")
    print(f"   • RLHF inicializado: {stats['rlhf_initialized']}")
    print(f"   • Alpha RLHF: {rag_rl_system.config['rlhf'].get('alpha', 0.1)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)