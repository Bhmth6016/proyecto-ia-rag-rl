from hf_client.client import query_hf_api

class ProductSummarizer:
    def __init__(self, model="summarization"):
        self.model = model
    
    def generate(self, product_data):
        summaries = []
        for product in product_data:
            prompt = self._create_prompt(product)
            summary = query_hf_api(prompt, model=self.model)
            summaries.append(summary)
        return summaries
    
    def _create_prompt(self, product):
        return f"""
        Generate a concise product summary in Spanish for:
        Name: {product['name']}
        Features: {product['features']}
        Reviews: {product['reviews']}
        """