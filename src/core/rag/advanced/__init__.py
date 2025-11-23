#src/core/rag/advanced/__init__

"""
Adapter para mantener compatibilidad con el código legacy.
Permite usar WorkingAdvancedRAGAgent con la interfaz antigua RAGAgent.
"""

from .WorkingRAGAgent import WorkingAdvancedRAGAgent, RAGResponse, RAGConfig


class RAGAgent:
    """
    Wrapper para mantener compatibilidad:
    - main.py utiliza ask()
    - cli.py utiliza ask()
    - main.py utiliza _save_conversation()
    """

    def __init__(self, products=None, enable_translation=False):
        # Configuración base – puedes personalizarla
        config = RAGConfig(
            enable_reranking=True,
            enable_rlhf=True,
            max_retrieved=50,
            max_final=5,
            domain="amazon"
        )

        # WorkingAdvancedRAGAgent no necesita productos aquí
        self.agent = WorkingAdvancedRAGAgent(config=config)

    def ask(self, query: str, user_id: str = "default") -> str:
        """
        Compatible con el método ask() esperado por main.py y cli.py.
        Debe devolver un string.
        """
        response: RAGResponse = self.agent.process_query(query, user_id)
        return response.answer

    def log_feedback(self, query: str, answer: str, rating: int, user_id: str = "default"):
        """Compatibilidad con RLHF legacy."""
        try:
            self.agent.log_feedback(query, answer, rating, user_id)
        except Exception:
            pass

    def _save_conversation(self, query: str, answer: str, rating: int = None):
        """
        Método llamado por main.py después de cada interacción.
        """
        if rating is not None:
            self.log_feedback(query, answer, rating)
