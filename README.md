# Amazon E-commerce Search System with RLHF and NER
A hybrid intelligent search system for e-commerce product discovery that combines semantic retrieval, Named Entity Recognition (NER), and Reinforcement Learning from Human Feedback (RLHF) with dynamic weight adaptation. The system operates entirely locally without cloud dependencies.
Key Features

Semantic Search: FAISS-based vector similarity search with sentence transformers
* NER Enhancement: Product attribute extraction and query intent detection
* RLHF with Dynamic Weights: Self-adjusting learning from user interactions
* Hybrid Ranking: Combines multiple complementary ranking signals
* Local Processing: Complete execution on local machine
* Reproducible Results: Consistent performance across multiple runs

# Project Structure
proyecto-ia-rag-rl/
├── src/
│   ├── unified_system.py              # Base search system
│   ├── unified_system_v2.py           # V2 with 4 ranking methods
│   ├── data/
│   │   ├── loader.py                  # Amazon dataset loader
│   │   ├── canonicalizer.py           # Product canonicalization
│   │   └── vector_store.py            # FAISS index manager
│   ├── ranking/
│   │   ├── baseline_ranker.py         # Basic FAISS ranking
│   │   ├── ner_enhanced_ranker.py     # NER-augmented ranking
│   │   └── rl_ranker_fixed.py         # RLHF with dynamic weights
│   ├── enrichment/
│   │   └── ner_zero_shot_optimized.py # NER attribute extraction
│   ├── query/
│   │   └── understanding.py           # Query intent analysis
│   └── features/
│       └── extractor.py               # Feature engineering
├── data/
│   ├── raw/                           # Raw Amazon product data
│   ├── cache/                         # Processed system caches
│   └── interactions/
│       ├── real_interactions.jsonl    # User click data
│       └── ground_truth_REAL.json     # Evaluation queries
├── results/                           # Experiment results
├── logs/                              # System logs
├── main.py                            # Main entry point
├── experimento_completo_4_metodos.py  # 4-method evaluation
├── extraer_ner_incremental.py         # Incremental NER processing
└── requirements.txt                   # Python dependencies

# Quick Start Guide
## 1. Prerequisites

Python 3.10+
16GB+ RAM (recommended for 90K products)
10GB+ disk space for models and data
CPU with AVX2 support (for FAISS)

## 2. Installation
### Clone repository
git clone <proyecto-ia-rag-rl>
cd proyecto-ia-rag-rl

### Create virtual environment
python -m venv ambiente
source ambiente/bin/activate  # Linux/Mac
### or
ambiente\Scripts\activate  # Windows

### Install dependencies
pip install -r requirements.txt

## 3. Data Preparation
Download Amazon product metadata from: https://amazon-reviews-2023.github.io
Place JSONL files in data/raw/:

meta_Video_Games_10000.jsonl
meta_Electronics_10000.jsonl
meta_Books_10000.jsonl
(and other categories)

### Expected format:
{
  "parent_asin": "B001234567",
  "title": "Product name",
  "description": ["Detailed description"],
  "features": ["Feature 1", "Feature 2"],
  "price": 29.99,
  "average_rating": 4.5,
  "rating_number": 1234,
  "categories": [["Electronics", "Video Games"]]
}

## 4. System Initialization
### Initialize system (processes 90K products, takes ~1.5 hours first time)
python main.py init

### Verify system integrity
python main.py stats

# Main Commands
### Build Search Index
python main.py init

Processes raw products:

* Canonicalization (title + description fusion)
* Embedding generation (all-MiniLM-L6-v2)
* FAISS index construction
* System cache creation

### Run Complete Experiment
python main.py experimento

Evaluates 4 ranking methods:

1. Baseline (FAISS semantic search)
2. NER-Enhanced (FAISS + attribute matching)
3. RLHF (FAISS + learned preferences)
4. Full Hybrid (FAISS + NER + RLHF)

### Interactive Search Mode
python main.py interactivo

Allows:

* Real-time product search
* User feedback collection (clicks)
* Interactive RLHF training

### Extract NER Attributes
python extraer_ner_incremental.py

Processes products for attribute extraction:

* Brand, color, size, material
* Category matching
* Incremental caching

# System Architecture
## 1. Four Ranking Methods
Method 1: Baseline

Pure semantic search with FAISS
Sentence transformer embeddings
L2 distance similarity

Method 2: NER-Enhanced

Baseline + attribute bonus scoring
Query intent detection
15% weight for attribute matches

Method 3: RLHF

Baseline + learned user preferences
30 features extracted per product
Dynamic weight adaptation

Method 4: Full Hybrid

Combines all three approaches
NER applied first, then RLHF
Best overall performance

## 2. RLHF Dynamic Weights
The system automatically learns which features matter:

# Feature types tracked
- semantic_match: Query-title overlap
- rating_value: Product ratings
- category_match: Category relevance
- specific_preferences: User-specific patterns

# Dynamic adjustment (every 5 feedbacks)
- Success tracking per feature type
- Automatic rebalancing
- Soft normalization

## 3. Reward Function
Position-based rewards:
- Position 1:     0.3 (obvious click)
- Position 2-3:   0.7 (good)
- Position 4-10:  1.2 (discovery)
- Position 10+:   1.5 (excellent discovery)
```

## Evaluation Metrics

### Primary Metrics

- **MRR (Mean Reciprocal Rank)**: Primary evaluation metric
- **P@5 (Precision at 5)**: Fraction of relevant items in top 5
- **R@5 (Recall at 5)**: Coverage of relevant items in top 5
- **NDCG@5**: Normalized Discounted Cumulative Gain

### Current Results
```
Method               MRR      P@5      R@5      NDCG@5
------------------------------------------------------
Baseline             0.5000  0.2400  0.5333  0.4634
NER Enhanced         0.5000  0.2400  0.5333  0.4634
RLHF                 0.6500  0.2000  0.5667  0.4866
Full Hybrid          0.7000  0.2400  0.6333  0.5896

Improvements vs Baseline:
- RLHF:        +30% (p=0.2080)
- Full Hybrid: +40% (p=0.1778)

### Dataset Statistics

Total products: 89,990
Categories: 9
Test queries: 5
Training queries: 13
Click interactions: 22
Relevant products: 31

# Configuration
## Main Settings (config/config.yaml)
data:
  limit: 100000                    # Product limit
  raw_path: "data/raw"
  cache_path: "data/cache"

models:
  embedding_model: "all-MiniLM-L6-v2"
  embedding_dim: 384

ranking:
  ner_weight: 0.15                 # NER bonus weight
  rlhf_learning_rate: 0.5
  match_rating_balance: 1.5

## Advanced Configuration
### In unified_system_v2.py
initialize_with_ner(
    limit=None,           # Process all products
    use_cache=True,       # Load from cache if available
    use_zero_shot=False   # Disable zero-shot NER (faster)
)

# Advanced Usage

1. Custom Ground Truth
Create data/interactions/ground_truth_REAL.json:
json{
  "queries": [
    {
      "query": "gaming laptop",
      "relevant_products": [
        "B08PRODUCT1",
        "B08PRODUCT2"
      ]
    }
  ]
}
2. Add User Interactions
Append to data/interactions/real_interactions.jsonl:
json{"interaction_type": "click", "timestamp": "2024-01-09T10:30:00", "context": {"query": "wireless mouse", "product_id": "B08MOUSE123", "position": 3}}
3. Retrain RLHF
python# In Python script
from src.unified_system_v2 import UnifiedSystemV2

system = UnifiedSystemV2.load_from_cache()
system.train_rlhf_with_queries(
    train_queries=["query1", "query2"],
    interactions_file="data/interactions/real_interactions.jsonl"
)
Troubleshooting
Cache Corruption Error
bash# Remove corrupted cache
rm data/cache/unified_system_v2.pkl

# Rebuild system
python main.py init
Memory Issues
python# Reduce batch size in data/canonicalizer.py
BATCH_SIZE = 1000  # Instead of 2000
Pickle Lambda Error
Already fixed in src/ranking/rl_ranker_fixed.py:
python# Helper function outside class
def _default_feature_stats():
    return {'hits': 0, 'total': 0}

# Inside class
self.feature_success = defaultdict(_default_feature_stats)
Performance Optimization
1. First Run (Cold Start)

Product loading: 5 seconds
Canonicalization: 60-90 minutes (90K products)
Index building: 2 seconds
NER processing: 1 second (1K products cached)
Total: ~1.5 hours

2. Subsequent Runs (Cache)

System loading: 2-3 seconds
Ready to search immediately

3. Experiment Execution

RLHF training: 1 second (22 interactions)
Evaluation (5 queries): 3-5 seconds
Total experiment: ~5 seconds (with cache)

Best Practices
1. Data Quality

Ensure products have titles and descriptions
Verify category fields are populated
Clean duplicate products before indexing

2. Evaluation

Use stratified train/test split
Maintain balanced query distribution
Collect diverse user interactions

3. RLHF Training

Minimum 20+ interactions recommended
Mix of positions (early and late clicks)
Regular retraining with new data

4. Production Deployment

Pre-build cache in staging environment
Monitor memory usage (16GB+ recommended)
Implement periodic cache refresh
Log all user interactions for retraining

Results Interpretation
Statistical Significance
Current p-values (0.1778 for Full Hybrid) indicate results are not statistically significant due to small test set (5 queries). To achieve p < 0.05:

Increase test queries to 30+
Collect 100+ click interactions
Ensure diverse query types

Why NER-Enhanced = Baseline?
Only 0.5% of products have NER attributes (436/89,990). To improve:
bash# Process all products
python extraer_ner_incremental.py


### RLHF Improvement Explanation

- Learned 30 features from 22 clicks
- Dynamic weights converged to prioritize semantic matches and ratings
- Position-based rewards emphasized discovery

## File Outputs

### Experiment Results

results/
├── experimento_4_metodos_YYYYMMDD_HHMMSS.json  # Detailed results
├── experimento_4_metodos_YYYYMMDD_HHMMSS.csv   # Metrics table
└── resumen_YYYYMMDD_HHMMSS.txt                 # Human-readable summary


### System Logs

logs/
└── experimento_YYYYMMDD_HHMMSS.log  # Complete execution log


### Cached Data

data/cache/
├── unified_system.pkl              # Base system
├── unified_system_v2.pkl           # V2 with RLHF
└── ner_cache_incremental.pkl       # NER attributes
Research Applications
This system serves as a foundation for research in:

E-commerce search and recommendation
Human-in-the-loop learning
Hybrid ranking systems
NER for product understanding
Dynamic weight adaptation
