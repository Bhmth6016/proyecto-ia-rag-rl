# src/core/llm/local_llm.py - Cliente LLM 100% local
import logging
import requests
import json
from typing import Optional, List, Dict, Any
from src.core.config import settings

logger = logging.getLogger(__name__)

class LocalLLMClient:
    """Cliente para modelos LLM locales (Ollama, llamafile, etc.)"""
    
    def __init__(self, 
             endpoint: Optional[str] = None, 
             model: Optional[str] = None,
             temperature: Optional[float] = None,
             timeout: Optional[int] = None,
             max_retries: int = 3):
        self.endpoint = endpoint or settings.LOCAL_LLM_ENDPOINT
        self.model = model or settings.LOCAL_LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LOCAL_LLM_TEMPERATURE
        self.timeout = timeout if timeout is not None else settings.LOCAL_LLM_TIMEOUT
        
        logger.info(f"🤖 LLM Local configurado: {self.model} en {self.endpoint}")
        logger.info(f"   • Temperature: {self.temperature}")
        logger.info(f"   • Timeout: {self.timeout}s")
    
    def check_availability(self) -> bool:
        """Verifica si Ollama está disponible en el endpoint."""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama no disponible en {self.endpoint}: {e}")
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            # Configurar payload para Ollama
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 512,  # Longitud máxima de respuesta
                    "repeat_penalty": 1.1
                }
            }
            
            logger.debug(f"Enviando petición a LLM local: {self.model}")
            
            # Enviar petición
            response = requests.post(
                f"{self.endpoint}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                
                # Limpiar respuesta si es necesario
                if content:
                    content = content.strip()
                    
                logger.debug(f"Respuesta LLM local recibida ({len(content)} chars)")
                return content
            else:
                error_msg = f"Error LLM local: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return self._fallback_response(prompt, error_msg)
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout en LLM local después de {self.timeout}s")
            return self._fallback_response(prompt, "Timeout del servidor")
        except Exception as e:
            logger.error(f"Error en LLM local: {e}")
            return self._fallback_response(prompt, str(e))
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        """Genera texto en streaming usando modelo local."""
        try:
            # Configurar mensajes
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            # Configurar payload para streaming
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            }
            
            # Enviar petición de streaming
            response = requests.post(
                f"{self.endpoint}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                json_str = line_text[6:]  # Remover 'data: '
                                if json_str.strip():
                                    chunk = json.loads(json_str)
                                    content = chunk.get('message', {}).get('content', '')
                                    if content:
                                        yield content
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.debug(f"Error procesando chunk: {e}")
            else:
                logger.error(f"Error en streaming: {response.status_code}")
                yield self._fallback_response(prompt, "Error en streaming")
                
        except Exception as e:
            logger.error(f"Error en streaming LLM: {e}")
            yield self._fallback_response(prompt, str(e))
    
    def _fallback_response(self, prompt: str, error: Optional[str] = None) -> str:

        if error:
            logger.info(f"Usando fallback debido a: {error}")
        
        return f"He procesado tu consulta sobre: '{prompt}'. Aquí tienes algunas recomendaciones basadas en nuestros productos."
    
    def list_available_models(self) -> List[str]:
        """Lista modelos disponibles en el endpoint Ollama."""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model.get('name') for model in data.get('models', [])]
                return models
        except Exception as e:
            logger.error(f"Error listando modelos: {e}")
        return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo actual."""
        try:
            response = requests.post(
                f"{self.endpoint}/api/show",
                json={"name": self.model},
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error obteniendo info del modelo: {e}")
        return {}
    
    @staticmethod
    def create_simple_response(query: str, products: List) -> str:
        """Crea respuesta simple sin LLM (más rápido)."""
        if not products:
            return f"No encontré productos específicos para '{query}'. ¿Podrías intentar con otros términos?"
        
        product_names = [getattr(p, 'title', f"Producto {i+1}") for i, p in enumerate(products[:3])]
        
        response_lines = [f"Para '{query}', te recomiendo:\n"]
        
        for i, name in enumerate(product_names, 1):
            # Truncar nombres largos
            if len(name) > 80:
                name = name[:77] + "..."
            response_lines.append(f"{i}. {name}")
        
        if len(products) > 3:
            response_lines.append(f"\n...y {len(products) - 3} productos más.")
        
        return "\n".join(response_lines)