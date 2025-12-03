import time
from typing import List, Dict, Optional
from profile_processor import ProfileProcessor
from searcher import JobSearcher


class RecommendationEngine:
    # Motor de recomendacion que combina procesamiento de perfil y busqueda FAISS
    
    def __init__(self, processed_data_dir: Optional[str] = None):
        # Inicializa el motor de recomendacion
        print("Inicializando Motor de Recomendación...")
        print("-" * 60)
        
        # Cargar componentes
        self.processor = ProfileProcessor()
        self.searcher = JobSearcher(processed_data_dir)
        
        print("-" * 60)
        print("OK - Motor de Recomendacion listo\n")
    
    def recomendar(self, perfil_texto: str, k: int = 10, verbose: bool = False) -> List[Dict]:
        # Recomienda las k ofertas mas relevantes para el perfil dado
        start_time = time.time()
        
        # Validar entrada
        if not perfil_texto or not isinstance(perfil_texto, str):
            raise ValueError("El perfil_texto debe ser un string no vacío")
        
        # 1. Procesar perfil de usuario
        if verbose:
            print("Procesando perfil...")
        
        embedding_time = time.time()
        perfil_embedding = self.processor.process_profile(perfil_texto)
        embedding_elapsed = time.time() - embedding_time
        
        # 2. Buscar ofertas similares
        if verbose:
            print("Buscando ofertas similares...")
        
        search_time = time.time()
        resultados = self.searcher.search(perfil_embedding, k=k)
        search_elapsed = time.time() - search_time
        
        # 3. Formatear resultados según especificación
        ofertas_formateadas = []
        for job in resultados:
            oferta = {
                'id': job['_global_index'],
                'title': job['title'],
                'description': job['description'],
                'description_preview': job['description'][:200] + '...' if len(job['description']) > 200 else job['description'],
                'score': round(job['similarity_score'], 4),
                'source': job.get('source', 'unknown'),
                'scraped_at': job.get('scraped_at', 'unknown'),
                '_source_file': job.get('_source_file', 'unknown'),
                'category': job.get('category', 'unknown')
            }
            ofertas_formateadas.append(oferta)
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nTiempos de ejecucion:")
            print(f"   - Embedding del perfil: {embedding_elapsed:.3f}s")
            print(f"   - Busqueda FAISS: {search_elapsed:.3f}s")
            print(f"   - Total: {total_time:.3f}s")
            print(f"OK - Encontradas {len(ofertas_formateadas)} ofertas relevantes\n")
        
        return ofertas_formateadas
    
    def get_statistics(self) -> Dict:
        # Retorna estadisticas del sistema de recomendacion
        return self.searcher.get_statistics()


def recomendar(perfil_texto: str, k: int = 10) -> List[Dict]:
    # Funcion standalone para recomendar ofertas
    engine = RecommendationEngine()
    return engine.recomendar(perfil_texto, k)


# Ejemplo de uso
if __name__ == "__main__":
    # Crear motor (se inicializa una sola vez)
    engine = RecommendationEngine()
    
    # Mostrar estadísticas
    print("="*70)
    print("ESTADÍSTICAS DEL SISTEMA")
    print("="*70)
    stats = engine.get_statistics()
    print(f"Total de ofertas indexadas: {stats['total_jobs']}")
    print(f"Dimensión de embeddings: {stats['embedding_dimension']}")
    print(f"\nDistribución por fuente:")
    for source, count in stats['sources'].items():
        categoria = source.replace('vectors_', '').replace('.pkl', '')
        print(f"  - {categoria}: {count} ofertas")
    
    # Ejemplo de perfil de usuario
    print("\n" + "="*70)
    print("EJEMPLO DE RECOMENDACIÓN")
    print("="*70)
    
    ejemplo_perfil = """
    Ingeniero de Software con 3 años de experiencia en desarrollo full stack.
    Experto en Python, Django, React y Node.js.
    Experiencia trabajando con bases de datos SQL (PostgreSQL, MySQL) y NoSQL (MongoDB).
    Conocimientos en Docker, Kubernetes y AWS.
    Familiarizado con metodologías ágiles (Scrum).
    Busco oportunidades en empresas de tecnología donde pueda aplicar 
    inteligencia artificial y machine learning.
    """
    
    print("\nPERFIL DEL USUARIO:")
    print("-" * 70)
    print(ejemplo_perfil.strip())
    print("-" * 70)
    
    # Obtener recomendaciones
    print("\nBuscando ofertas relevantes...")
    ofertas = engine.recomendar(ejemplo_perfil, k=5, verbose=True)
    
    # Mostrar resultados
    print("="*70)
    print("TOP 5 OFERTAS RECOMENDADAS")
    print("="*70)
    
    for i, oferta in enumerate(ofertas, 1):
        print(f"\n{i}. {oferta['title']}")
        print(f"   Score de similitud: {oferta['score']:.4f}")
        print(f"   Categoria: {oferta['_source_file'].replace('vectors_', '').replace('.pkl', '')}")
        print(f"   Preview: {oferta['description_preview']}")
        print(f"   ID: {oferta['id']}")
    
    print("\n" + "="*70)
    print("OK - Sistema de recomendacion funcionando correctamente")
    print("="*70)
