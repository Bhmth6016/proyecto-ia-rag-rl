# src/data/loader.py
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from tqdm import tqdm
from src.core.data.product import Product  # Assuming we have the Product model

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(
        self,
        raw_dir: Union[str, Path] = None,
        processed_dir: Union[str, Path] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize the data loader with directories and caching options.
        
        Args:
            raw_dir: Directory containing raw data files
            processed_dir: Directory for processed data
            cache_enabled: Whether to use caching for processed data
        """
        self.base_dir = Path(__file__).parent.parent.parent.resolve()
        self.raw_dir = Path(raw_dir) if raw_dir else (self.base_dir / "data" / "raw")
        self.processed_dir = Path(processed_dir) if processed_dir else (self.base_dir / "data" / "processed")
        self.cache_enabled = cache_enabled
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def raw_files(self) -> List[Path]:
        """Get all raw data files in the raw directory"""
        return list(self.raw_dir.glob("*.json")) + list(self.raw_dir.glob("*.jsonl"))

    def _get_cache_file(self, source_file: Path) -> Path:
        """Get the cache file path for a given source file"""
        return self.processed_dir / f"{source_file.stem}_processed.pkl"

    def _needs_processing(self, raw_file: Path, cache_file: Path) -> bool:
        """Check if a file needs processing (either no cache or cache is stale)"""
        if not self.cache_enabled:
            return True
        return not cache_file.exists() or raw_file.stat().st_mtime > cache_file.stat().st_mtime

    def _process_raw_file(self, file_path: Path) -> List[Product]:
        """
        Process a single raw file (JSON or JSONL) into Product objects
        
        Args:
            file_path: Path to the raw data file
            
        Returns:
            List of validated Product objects
        """
        products = []
        error_count = 0
        total_lines = 0
        
        # Count lines for progress bar (JSONL only)
        if file_path.suffix == ".jsonl":
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
        
        try:
            # Read the file based on its type
            if file_path.suffix == ".json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("JSON file should contain an array of products")
                items = data
            else:  # JSONL
                items = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, total=total_lines, desc=f"Processing {file_path.name}"):
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            error_count += 1
                            continue
            
            # Process each item
            for item in items:
                try:
                    # Convert to Product model (validation happens here)
                    product = Product.from_raw(item)
                    products.append(product)
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Invalid product skipped: {str(e)}")
            
            if error_count > 0:
                logger.warning(
                    f"{file_path.name}: {error_count} errors out of {len(items)} items "
                    f"({(1 - error_count/max(1, len(items)))*100:.1f}% success)"
                )
            
            return products
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {str(e)}")
            raise

    def _save_to_cache(self, products: List[Product], cache_file: Path) -> None:
        """Save processed products to cache file"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(products, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to save cache {cache_file.name}: {str(e)}")
            raise

    def _load_from_cache(self, cache_file: Path) -> List[Product]:
        """Load processed products from cache file"""
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache {cache_file.name}: {str(e)}")
            raise

    def load_single_file(self, file_path: Path, use_cache: bool = True) -> List[Product]:
        """
        Load and process a single file, with optional caching
        
        Args:
            file_path: Path to the data file
            use_cache: Whether to use cached version if available
            
        Returns:
            List of Product objects
        """
        cache_file = self._get_cache_file(file_path)
        
        if use_cache and not self._needs_processing(file_path, cache_file):
            try:
                return self._load_from_cache(cache_file)
            except Exception:
                logger.warning("Cache load failed, reprocessing file")
                
        # Process the file if cache is disabled or stale
        products = self._process_raw_file(file_path)
        
        # Save to cache if processing was successful
        if products and use_cache:
            self._save_to_cache(products, cache_file)
            
        return products

    def load_data(self, use_cache: bool = True) -> List[Product]:
        """
        Load all data from raw directory
        
        Args:
            use_cache: Whether to use cached versions if available
            
        Returns:
            Combined list of all Product objects from all files
        """
        all_products = []
        total_files = 0
        total_errors = 0
        
        logger.info(f"Loading data from {self.raw_dir}")
        
        for raw_file in self.raw_files:
            total_files += 1
            try:
                products = self.load_single_file(raw_file, use_cache=use_cache)
                all_products.extend(products)
                logger.info(f"Loaded {len(products)} products from {raw_file.name}")
            except Exception as e:
                total_errors += 1
                logger.error(f"Failed to load {raw_file.name}: {str(e)}")
                continue
                
        logger.info(
            f"Loaded {len(all_products)} products from {total_files} files "
            f"({total_errors} files failed)"
        )
        
        return all_products

    def export_to_json(self, products: List[Product], output_file: Path) -> None:
        """
        Export processed products to JSON file
        
        Args:
            products: List of Product objects to export
            output_file: Path to output JSON file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(
                    [product.to_dict() for product in products],
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str
                )
            logger.info(f"Exported {len(products)} products to {output_file}")
        except Exception as e:
            logger.error(f"Failed to export to {output_file}: {str(e)}")
            raise

    def get_latest_raw_file(self) -> Optional[Path]:
        """Get the most recently modified raw data file"""
        if not self.raw_files:
            return None
        return max(self.raw_files, key=lambda f: f.stat().st_mtime)

    def clear_cache(self) -> int:
        """Clear all cached processed files and return count of files deleted"""
        cache_files = list(self.processed_dir.glob("*_processed.pkl"))
        deleted = 0
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete {cache_file.name}: {str(e)}")
                
        logger.info(f"Deleted {deleted} cache files")
        return deleted