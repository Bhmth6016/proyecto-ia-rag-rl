import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Mapeo de palabras clave
# ------------------------------------------------------------------
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "backpack": ["mochila", "backpack", "bagpack", "laptop bag"],
    "headphones": ["auriculares", "headphones", "headset", "earbuds"],
    "speaker": ["altavoz", "speaker", "bluetooth speaker", "portable speaker"],
    "keyboard": ["teclado", "keyboard", "mechanical keyboard"],
    "mouse": ["ratón", "mouse", "wireless mouse"],
    "monitor": ["monitor", "pantalla", "screen", "display"],
    "camera": ["cámara", "camera", "webcam", "dslr"],
    "home_appliance": ["aspiradora", "vacuum", "microondas", "microwave"],
}

_TAG_KEYWORDS: Dict[str, List[str]] = {
    "waterproof": ["waterproof", "water resistant", "resistente al agua"],
    "wireless": ["wireless", "bluetooth", "inalámbrico", "wifi"],
    "portable": ["portable", "pocket", "ligero", "lightweight"],
    "gaming": ["gaming", "gamer", "rgb", "for gaming"],
    "travel": ["travel", "viaje", "suitcase", "carry-on"],
    "usb-c": ["usb-c", "type-c", "usb type c"],
    "noise-cancelling": ["noise cancelling", "noise reduction", "anc"],
    "fast-charging": ["fast charging", "quick charge", "carga rápida"],
}

class DataLoader:
    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    @staticmethod
    def _infer_product_type(title: str, specs: Dict[str, Any]) -> Optional[str]:
        text = (title or "").lower() + " " + " ".join(specs.values()).lower()
        for ptype, kw_list in _CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in kw_list):
                return ptype
        return None

    @staticmethod
    def _extract_tags(title: str, specs: Dict[str, Any]) -> List[str]:
        text = (title or "").lower() + " " + " ".join(specs.values()).lower()
        tags = []
        for tag, kw_list in _TAG_KEYWORDS.items():
            if any(kw in text for kw in kw_list):
                tags.append(tag)
        return tags

    def load_and_clean_jsonl(self, file_path: Path) -> List[Dict]:
        """Carga y limpia un solo archivo JSONL"""
        cleaned = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    cleaned.append(self._clean_product(data))
                except json.JSONDecodeError:
                    continue
        return cleaned

    def _clean_product(self, raw: Dict) -> Dict:
        """Estandariza un producto individual"""
        return {
            "title": raw.get("title", "").strip(),
            "price": float(raw.get("price", 0)),
            "product_type": self._infer_product_type(raw.get("title", ""), raw.get("details", {}).get("specifications", {})),
            "tags": self._extract_tags(raw.get("title", ""), raw.get("details", {}).get("specifications", {})),
        }

    def create_unified_json(self, output_file: Path) -> bool:
        """Procesa todos los JSONL y crea un JSON unificado"""
        all_products = []
        for jsonl_file in self.raw_dir.glob("*.jsonl"):
            all_products.extend(self.load_and_clean_jsonl(jsonl_file))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_products, f, indent=2)
        
        return True

# Ejemplo de uso
if __name__ == "__main__":
    raw_dir = Path("path/to/raw_dir")
    processed_dir = Path("path/to/processed_dir")
    output_file = processed_dir / "unified_products.json"

    data_loader = DataLoader(raw_dir, processed_dir)
    data_loader.create_unified_json(output_file)