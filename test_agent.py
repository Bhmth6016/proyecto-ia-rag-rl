# test_agent.py
import sys
from pathlib import Path

# Add src to Python path so imports work
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.data.loader import DataLoader
from src.core.rag.advanced.agent import RAGAgent

def main():
    print("Loading data...")
    loader = DataLoader()
    products = loader.load_data(use_cache=True)[:1000]  # Smaller subset
    print(f"Loaded {len(products)} products")

    print("Initializing agent...")
    agent = RAGAgent(products=products)

    print("Starting chat loop...")
    agent.chat_loop()

if __name__ == "__main__":
    main()