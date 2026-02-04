# Amazon E-commerce Search System with RLHF and NER

 A hybrid intelligent search system for e-commerce product discovery that combines semantic retrieval, Named Entity Recognition (NER), and Reinforcement Learning from Human Feedback (RLHF) with dynamic weight adaptation. The system operates entirely locally without cloud dependencies.
 
## Key Features

* Semantic Search: FAISS-based vector similarity search with sentence transformers 
* NER Enhancement: Product attribute extraction and query intent detection RLHF with 
* Dynamic Weights: Self-adjusting learning from user interactions 
* Hybrid Ranking: Combines multiple complementary ranking signals 
* Review-Based Training: Generate training data from Amazon reviews 
* Local Processing: Complete execution on local machine 
* Reproducible Results: Consistent performance across multiple runs 

## Project Structure
```

proyecto-ia-rag-rl/
├── src/
│   ├── unified_system.py              # Base search system
│   ├── unified_system_v2.py           # V2 with 4 ranking methods
│   ├── data/
│   │   ├── loader.py                  # Amazon dataset loader
│   │   ├── canonicalizer.py           # Product canonicalization
│   │   ├── cache_manager.py           # Cache management
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
│   ├── raw/                           # Raw Amazon product metadata
│   ├── reviews/                       # Amazon review data
│   ├── cache/                         # Processed system caches
│   ├── rlhf_pairs/                    # Generated RLHF training pairs
│   └── interactions/
│       ├── real_interactions.jsonl    # User click data
│       ├── rlhf_interactions_from_reviews.jsonl  # Review-based training
│       ├── ground_truth_REAL.json     # Evaluation queries (manual)
│       └── ground_truth_from_reviews.json  # Evaluation queries (reviews)
├── results/                           # Experiment results
├── logs/                              # System logs
├── main.py                            # Main entry point
├── experimento_completo_4_metodos.py  # 4-method evaluation
├── extraer_ner_incremental.py         # Incremental NER processing
├── generate_rlhf_pairs_from_reviews.py  # Generate training pairs
├── integrate_rlhf_pairs.py            # Integrate pairs into system
├── pipeline_reviews_to_rlhf.py        # Complete review pipeline
├── regenerate_interactions.py         # Fix old interactions
├── reviews_from_products_10000.py     # Filter reviews by products
├── sistema_interactivo.py             # Interactive search interface
└── requirements.txt                   # Python dependencies
```

## Quick Start Guide 

### 1. Prerequisites 
Python 3.10+ 16GB+ RAM (recommended for 90K products) 10GB+ disk space for models and data CPU with AVX2 support (for FAISS) 

### 2. Installation 
Clone repository 
git clone <proyecto-ia-rag-rl> 
cd proyecto-ia-rag-rl 

#### Create virtual environment 
python -m venv ambiente 
source ambiente/bin/activate # Linux/Mac 

#### or 
ambiente\Scripts\activate # Windows 

#### Install dependencies 
pip install -r requirements.txt

### 3. Data Preparation

Download Amazon product metadata and reviews from: https://amazon-reviews-2023.github.io

Place JSONL files in appropriate directories:

**Products (data/raw/):**
* meta_Video_Games.jsonl 
* meta_Electronics.jsonl 
* meta_Books.jsonl 
* meta_Automotive.jsonl  
* meta_Beauty_and_Personal_Care.jsonl 
* meta_Clothing_Shoes_and_Jewelry.jsonl 
* meta_Home_and_Kitchen.jsonl
* meta_Sports_and_Outdoors.jsonl 
* meta_Toys_and_Games.jsonl

**Reviews (data/reviews/):**
* Video_Games.jsonl 
* Electronics_10000.jsonl 
* Books_10000.jsonl (and other categories) 

#### Expected product format: 
json{ "parent_asin": "B001234567", 
"title": "Product name", 
"description": ["Detailed description"], 
"features": ["Feature 1", "Feature 2"], 
"price": 29.99, "average_rating": 4.5, 
"rating_number": 1234, 
"main_category": "Video Games", 
"imageURL": "https://..." 
}

#### Expected review format: 
json{ "parent_asin": "B001234567", 
"rating": 5.0, "helpful_vote": 12, 
"verified_purchase": true, 
"text": "Great product!" 
} 

### 4. System Initialization 
#### Initialize system (processes products, takes ~1.5 hours first time) 
python main.py init 
#### Verify system integrity 
python main.py stats 

## Main Commands 
### Build Search Index 
python main.py init 
Processes raw products: 
* Canonicalization (title + description fusion) 
* Embedding generation (all-MiniLM-L6-v2) 
* FAISS index construction 
* System cache creation 
### Generate Training Data from Reviews 
#### Complete pipeline (automated) 
python pipeline_reviews_to_rlhf.py 
This will: 
1. Auto-detect all available categories 
2. Generate RLHF pairs from reviews 
3. Integrate pairs into system 
4. Create ground truth 
#### Manual steps (if needed): 
##### Step 1: Generate pairs for all categories 
python generate_rlhf_pairs_from_reviews.py 
##### Step 2: Integrate pairs 
python integrate_rlhf_pairs.py 
#### Collect Real User Interactions 
python main.py interactivo 
Interactive search mode allows: 
* Real-time product search 
* User feedback collection (clicks) 
* Interactive RLHF training 

Important: Only products with real Amazon IDs (starting with 'B') are shown. 
#### Run Complete Experiment 
python main.py experimento 
Evaluates 4 ranking methods: 
1. Baseline (FAISS semantic search) 
2. NER-Enhanced (FAISS + attribute matching) 
3. RLHF (FAISS + learned preferences) 
4. Full Hybrid (FAISS + NER + RLHF) 
#### Extract NER Attributes 
python extraer_ner_incremental.py 
Processes products for attribute extraction: 
* Brand, color, size, material 
* Category matching 
* Incremental caching (processes only new/changed products) 
#### Fix Old Interactions 
python regenerate_interactions.py 
Updates old interaction files to use current product IDs. 
### System Architecture 
#### 1. Four Ranking Methods 
##### Method 1: Baseline 
* Pure semantic search with FAISS 
* Sentence transformer embeddings 
* L2 distance similarity 
##### Method 2: NER-Enhanced 
* Baseline + attribute bonus scoring 
* Query intent detection 
* 15% weight for attribute matches 
##### Method 3: RLHF 
* Baseline + learned user preferences 
* 30 features extracted per product 
* Dynamic weight adaptation 
##### Method 4: Full Hybrid 
* Combines all three approaches 
* NER applied first, then RLHF 
* Best overall performance 
#### 2. RLHF Dynamic Weights 
The system automatically learns which features matter: 
#### Feature types tracked 
- semantic_match: Query-title overlap 
- rating_value: Product ratings 
- category_match: Category relevance 
- specific_preferences: User-specific patterns 
##### Dynamic adjustment (every 5 feedbacks) 
- Success tracking per feature type 
- Automatic rebalancing 
- Soft normalization 
#### 3. Review-Based Reward Function 
Product Reward Calculation: pythonreward = ( α * rating_normalized + # 0.4 weight β * helpful_votes_score + # 0.3 weight γ * verified_purchase_ratio + # 0.2 weight δ * recency_score # 0.1 weight )
**Position-based rewards (for clicks):**
- Position 1:     0.3 (obvious click)
- Position 2-3:   0.7 (good)
- Position 4-10:  1.2 (discovery)
- Position 10+:   1.5 (excellent discovery)

### 4. Data Processing Pipeline
```
Raw Products → Canonicalization → Embeddings → FAISS Index 
                    ↓ 
              NER Extraction 
                    ↓ 
                Reviews → Aggregate by Product → Calculate Rewards → RLHF Pairs 
                                                                            ↓ 
                                                                      Ground Truth 
                                                                      
```
## Evaluation Metrics 
### Primary Metrics 
#### * MRR (Mean Reciprocal Rank): Primary evaluation metric 
#### * P@5 (Precision at 5): Fraction of relevant items in top 5 
#### * R@5 (Recall at 5): Coverage of relevant items in top 5 
#### * NDCG@5: Normalized Discounted Cumulative Gain 

Statistical Significance Testing 
* t-test (paired samples) 
* Cohen's d (effect size) 
* p-value < 0.05 for significance 

### Configuration 
Main Settings (config/config.yaml) 
data: 
limit: 100000 # Product limit 
raw_path: "data/raw" 
cache_path: "data/cache" 

models: 
embedding_model: "all-MiniLM-L6-v2" 
embedding_dim: 384 

ranking: 
ner_weight: 0.15 # NER bonus weight 
rlhf_learning_rate: 0.5 
match_rating_balance: 1.5 

reviews: 
min_reviews: 5 # Min reviews per product 
pairs_per_query: 3 # Training pairs per query 

#### Advanced Configuration 
In unified_system_v2.py:
pythoninitialize_with_ner(
    limit=None,           # Process all products
    use_cache=True,       # Load from cache if available
    use_zero_shot=False   # Disable zero-shot NER (faster)
)

In generate_rlhf_pairs_from_reviews.py:
pythongenerator = RLHFPairGenerator(
    min_reviews=5,           # Minimum reviews per product
    pairs_per_query=3        # Pairs generated per query
)
Advanced Usage
1. Custom Ground Truth (Manual)
Create data/interactions/ground_truth_REAL.json:
json{
  "gaming laptop": ["B08PRODUCT1", "B08PRODUCT2"],
  "wireless mouse": ["B08MOUSE123"]
}
2. Add User Interactions
Append to data/interactions/real_interactions.jsonl:
json{"timestamp": "2024-01-09T10:30:00", "session_id": "session_123", "interaction_type": "click", "context": {"query": "wireless mouse", "product_id": "B08MOUSE123", "position": 3, "product_title": "Logitech Wireless Mouse", "has_real_id": true, "is_relevant": true}}
3. Retrain RLHF
python# In Python script
from src.unified_system_v2 import UnifiedSystemV2

system = UnifiedSystemV2.load_from_cache()
system.train_rlhf_with_queries(
    train_queries=["query1", "query2"],
    interactions_file="data/interactions/real_interactions.jsonl"
)
4. Process Specific Categories
python# In generate_rlhf_pairs_from_reviews.py
generator.process_category(
    category="Video_Games",
    limit_products=10000,
    limit_reviews=100000
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
ID Mismatch Issues
If you encounter ID mismatches between old and new data:
bash# Regenerate interactions with current IDs
python regenerate_interactions.py
Missing Review Files
bash# Filter reviews to match sampled products
python reviews_from_products_10000.py


## Performance Optimization

### 1. First Run (Cold Start)

- Product loading: 5 seconds
- Canonicalization: 60-90 minutes (90K products)
- Index building: 2 seconds
- NER processing: 1 second (with cache)
- **Total: ~1.5 hours**

### 2. Subsequent Runs (Cache)

- System loading: 2-3 seconds
- Ready to search immediately

### 3. Review Processing

- RLHF pair generation (all categories): 5-10 minutes
- Integration: 1-2 seconds

### 4. Experiment Execution

- RLHF training: 1 second (with cached data)
- Evaluation (5 queries): 3-5 seconds
- **Total experiment: ~5 seconds (with cache)**

## Best Practices

### 1. Data Quality

- Ensure products have titles and descriptions
- Verify `parent_asin` fields are populated
- Clean duplicate products before indexing
- Match review files with product files

### 2. Review-Based Training

- Use products with 5+ reviews
- Verify `parent_asin` consistency across files
- Check for helpful votes and verified purchases
- Balance categories in training data

### 3. Evaluation

- Use stratified train/test split (75/25)
- Maintain balanced query distribution
- Collect diverse user interactions
- Test with 30+ queries for statistical significance

### 4. RLHF Training

- **Minimum 20+ interactions** recommended
- Mix of positions (early and late clicks)
- Regular retraining with new data
- Use both manual clicks and review-based pairs

### 5. Production Deployment

- Pre-build cache in staging environment
- Monitor memory usage (16GB+ recommended)
- Implement periodic cache refresh
- Log all user interactions for retraining
- Use incremental NER processing for updates

## File Outputs

### Experiment Results
```
results/
├── experimento_4_metodos_YYYYMMDD_HHMMSS.json  # Detailed results
├── experimento_4_metodos_YYYYMMDD_HHMMSS.csv   # Metrics table
└── resumen_YYYYMMDD_HHMMSS.txt                 # Human-readable summary
```

### System Logs
```
logs/
└── experimento_YYYYMMDD_HHMMSS.log  # Complete execution log
```

### Cached Data
```
data/cache/
├── unified_system.pkl              # Base system
├── unified_system_v2.pkl           # V2 with RLHF
├── ner_cache_incremental.pkl       # NER attributes
└── canonical/                      # Canonicalized products
```

### Training Data
```
data/rlhf_pairs/
└── rlhf_pairs_*.jsonl             # Training pairs per category

data/interactions/
├── rlhf_interactions_from_reviews.jsonl  # Review-based training
└── ground_truth_from_reviews.json        # Evaluation queries
```
## Research Applications

This system serves as a foundation for research in:

- E-commerce search and recommendation
- Human-in-the-loop learning
- Hybrid ranking systems
- NER for product understanding
- Dynamic weight adaptation
- Review-based implicit feedback
- Multi-signal ranking fusion

## Data Flow Diagram
```

                    ┌─────────────────┐
                    │  Raw Products   │
                    │  (90K+ items)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Canonicalization│
                    │   + Embeddings  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────────┐    │    ┌────────▼────────┐
     │  FAISS Index    │    │    │  NER Extraction │
     │  (Semantic)     │    │    │  (Attributes)   │
     └────────┬────────┘    │    └────────┬────────┘
              │              │              │
              │    ┌─────────▼────────┐    │
              │    │  Amazon Reviews  │    │
              │    │  (Aggregated)    │    │
              │    └─────────┬────────┘    │
              │              │              │
              │    ┌─────────▼────────┐    │
              │    │  RLHF Pairs      │    │
              │    │  (chosen/reject) │    │
              │    └─────────┬────────┘    │
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Hybrid Ranking │
                    │  (4 Methods)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Search Results │
                    └─────────────────┘
```