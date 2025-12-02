import re
from typing import Union, List
import numpy as np
from sentence_transformers import SentenceTransformer


class ProfileProcessor:
    # Procesador de perfiles de usuario que genera embeddings
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        # Inicializa el procesador con el modelo de embeddings
        print(f"Cargando modelo: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("OK - Modelo cargado exitosamente")
    
    def clean_text(self, text: str) -> str:
        # Normaliza texto: minusculas, sin caracteres especiales, espacios normalizados
        if not isinstance(text, str):
            return ""
        
        # Convertir a minúsculas
        normalized = text.lower()
        
        # Remover saltos de línea, tabs
        normalized = re.sub(r'[\n\t\r]', ' ', normalized)
        
        # Remover caracteres especiales (mantener letras, números, acentos)
        normalized = re.sub(r'[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ\s]', '', normalized)
        
        # Normalizar espacios múltiples
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def process_profile(self, profile_text: str) -> np.ndarray:
        # Procesa un perfil de usuario y genera su embedding
        if not profile_text or not isinstance(profile_text, str):
            raise ValueError("El texto del perfil no puede estar vacío")
        
        # Limpiar texto
        cleaned = self.clean_text(profile_text)
        
        if not cleaned:
            raise ValueError("El texto del perfil no contiene contenido válido después de limpieza")
        
        # Generar embedding
        embedding = self.model.encode(cleaned, show_progress_bar=False)
        
        return embedding
    
    def process_profiles_batch(self, profiles: List[str]) -> np.ndarray:
        # Procesa multiples perfiles en lote (mas eficiente)
        if not profiles:
            raise ValueError("La lista de perfiles no puede estar vacía")
        
        # Limpiar todos los textos
        cleaned_profiles = [self.clean_text(p) for p in profiles]
        
        # Validar que haya contenido válido
        valid_profiles = [p for p in cleaned_profiles if p]
        if not valid_profiles:
            raise ValueError("Ningún perfil contiene contenido válido después de limpieza")
        
        # Generar embeddings en lote
        embeddings = self.model.encode(valid_profiles, show_progress_bar=True)
        
        return embeddings


# Ejemplo de uso
if __name__ == "__main__":
    # Crear procesador
    processor = ProfileProcessor()
    
    # Ejemplo de perfil de usuario
    ejemplo_cv = """
    Desarrollador Full Stack con 3 años de experiencia.
    Experto en Python, Django, React y Node.js.
    Experiencia en bases de datos SQL y NoSQL.
    Conocimientos en machine learning y data science.
    Busco oportunidades en empresas innovadoras.
    """
    
    print("\nProcesando perfil de ejemplo...")
    embedding = processor.process_profile(ejemplo_cv)
    
    print(f"OK - Embedding generado")
    print(f"  - Dimensión: {embedding.shape}")
    print(f"  - Primeros 5 valores: {embedding[:5]}")
