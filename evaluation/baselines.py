import random
import pandas as pd
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# Agregar la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PLN.searcher import JobSearcher

class Baselines:
    def __init__(self, processed_data_dir: str = None):
        self.searcher = JobSearcher(processed_data_dir)
        self.jobs = self.searcher.job_metadata
        self.df = pd.DataFrame(self.jobs)
        
        # Pre-calcular matriz TF-IDF para el baseline TF-IDF
        print("Entrenando vectorizador TF-IDF para baseline...")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        # Usar cleaned_text si está disponible, sino description
        texts = [job.get('cleaned_text', job.get('description', '')) for job in self.jobs]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print("Vectorizador TF-IDF listo.")

    def random_recommendation(self, k: int = 10) -> List[Dict]:
        """Baseline 1: Aleatorio"""
        return random.sample(self.jobs, k)

    def popularity_recommendation(self, k: int = 10) -> List[Dict]:
        """Baseline 2: Popularidad (Más recientes)"""
        # Ordenar por scraped_at descendente
        sorted_jobs = self.df.sort_values('scraped_at', ascending=False).head(k)
        return sorted_jobs.to_dict('records')

    def tfidf_recommendation(self, profile_text: str, k: int = 10) -> List[Dict]:
        """Baseline 3: TF-IDF"""
        profile_vector = self.vectorizer.transform([profile_text])
        similarities = cosine_similarity(profile_vector, self.tfidf_matrix).flatten()
        # Obtener los índices top k
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append(self.jobs[idx])
        return results
