import pickle
import os
import glob
from typing import List, Dict, Tuple
import numpy as np
import faiss


class JobSearcher:
    # Motor de busqueda de ofertas laborales usando FAISS
    
    def __init__(self, processed_data_dir: str = None):
        # Inicializa el buscador y carga todos los embeddings
        if processed_data_dir is None:
            # Obtener ruta relativa desde este archivo
            current_dir = os.path.dirname(os.path.abspath(__file__))
            processed_data_dir = os.path.join(
                os.path.dirname(current_dir), 
                'dataset', 
                'processed'
            )
        
        self.processed_data_dir = processed_data_dir
        self.index = None
        self.job_metadata = []
        self.embedding_dim = None
        
        print(f"Cargando datos desde: {self.processed_data_dir}")
        self._load_all_data()
        self._build_index()
        print(f"OK - Indice FAISS creado con {len(self.job_metadata)} ofertas")
    
    def _load_all_data(self):
        # Carga todos los archivos .pkl y combina metadata y embeddings
        pkl_files = glob.glob(os.path.join(self.processed_data_dir, "vectors_*.pkl"))
        
        if not pkl_files:
            raise FileNotFoundError(
                f"No se encontraron archivos .pkl en {self.processed_data_dir}. "
                f"Ejecuta process_embeddings.py primero."
            )
        
        print(f"Encontrados {len(pkl_files)} archivos de embeddings")
        
        all_embeddings = []
        
        for pkl_file in sorted(pkl_files):
            filename = os.path.basename(pkl_file)
            print(f"  Cargando {filename}...")
            
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            metadata = data['metadata']
            embeddings = data['embeddings']
            
            # Agregar índice global a cada oferta
            start_idx = len(self.job_metadata)
            for i, job in enumerate(metadata):
                job['_global_index'] = start_idx + i
                job['_source_file'] = filename
            
            self.job_metadata.extend(metadata)
            all_embeddings.append(embeddings)
        
        # Combinar todos los embeddings en un solo array
        self.all_embeddings = np.vstack(all_embeddings).astype('float32')
        self.embedding_dim = self.all_embeddings.shape[1]
        
        print(f"OK - Cargadas {len(self.job_metadata)} ofertas")
        print(f"  Dimensión de embeddings: {self.embedding_dim}")
    
    def _build_index(self):
        # Construye el indice FAISS para busqueda rapida (IndexFlatIP)
        # Normalizar embeddings para usar producto interno como similitud coseno
        faiss.normalize_L2(self.all_embeddings)
        
        # Crear índice (Inner Product = cosine similarity cuando vectores normalizados)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Agregar vectores al índice
        self.index.add(self.all_embeddings)
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict]:
        # Busca las k ofertas mas similares al embedding de consulta
        if self.index is None:
            raise RuntimeError("Índice no inicializado. Llama a _build_index() primero.")
        
        # Asegurar que el query es float32 y 2D
        query = np.array(query_embedding, dtype='float32')
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Normalizar para similitud coseno
        faiss.normalize_L2(query)
        
        # Limitar k al número de ofertas disponibles
        k = min(k, len(self.job_metadata))
        
        # Buscar en el índice
        scores, indices = self.index.search(query, k)
        
        # Construir resultados
        results = []
        for score, idx in zip(scores[0], indices[0]):
            job = self.job_metadata[idx].copy()
            job['similarity_score'] = float(score)
            results.append(job)
        
        return results
    
    def get_job_by_index(self, index: int) -> Dict:
        # Obtiene una oferta por su indice global
        if 0 <= index < len(self.job_metadata):
            return self.job_metadata[index]
        raise IndexError(f"Índice {index} fuera de rango (0-{len(self.job_metadata)-1})")
    
    def get_statistics(self) -> Dict:
        # Retorna estadisticas del dataset indexado
        sources = {}
        for job in self.job_metadata:
            source_file = job.get('_source_file', 'unknown')
            sources[source_file] = sources.get(source_file, 0) + 1
        
        return {
            'total_jobs': len(self.job_metadata),
            'embedding_dimension': self.embedding_dim,
            'sources': sources,
            'index_type': type(self.index).__name__
        }


# Ejemplo de uso
if __name__ == "__main__":
    # Crear buscador (carga automáticamente todos los datos)
    searcher = JobSearcher()
    
    # Mostrar estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS DEL ÍNDICE")
    print("="*60)
    stats = searcher.get_statistics()
    for key, value in stats.items():
        if key == 'sources':
            print(f"\n{key}:")
            for source, count in value.items():
                print(f"  - {source}: {count} ofertas")
        else:
            print(f"{key}: {value}")
    
    # Ejemplo de búsqueda con embedding aleatorio
    print("\n" + "="*60)
    print("PRUEBA DE BÚSQUEDA")
    print("="*60)
    print("Generando embedding de prueba...")
    
    # Usar un embedding real del dataset para probar
    test_embedding = searcher.all_embeddings[0]
    results = searcher.search(test_embedding, k=5)
    
    print(f"\nTop 5 resultados más similares:")
    for i, job in enumerate(results, 1):
        print(f"\n{i}. {job['title'][:60]}...")
        print(f"   Score: {job['similarity_score']:.4f}")
        print(f"   Fuente: {job['_source_file']}")
