import json
import os
import sys
import pandas as pd
import numpy as np

# Agregar la raíz del proyecto al path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'PLN'))

from PLN.recommender import RecommendationEngine
from evaluation.baselines import Baselines

# Funciones de métricas
def precision_at_k(recomendados: list, relevantes: set, k: int) -> float:
    top_k = recomendados[:k]
    relevantes_en_top_k = len([x for x in top_k if x in relevantes])
    return relevantes_en_top_k / k

def recall_at_k(recomendados: list, relevantes: set, k: int) -> float:
    top_k = recomendados[:k]
    relevantes_en_top_k = len([x for x in top_k if x in relevantes])
    return relevantes_en_top_k / len(relevantes) if relevantes else 0

def mrr(recomendados: list, relevantes: set) -> float:
    for i, item in enumerate(recomendados, 1):
        if item in relevantes:
            return 1.0 / i
    return 0.0

def hit_rate_at_k(recomendados: list, relevantes: set, k: int) -> float:
    top_k = recomendados[:k]
    return 1.0 if any(x in relevantes for x in top_k) else 0.0

def evaluate_model(model_name, predictions, ground_truth, k=10):
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    hit_rates = []
    
    for profile_id, recommended_ids in predictions.items():
        if profile_id not in ground_truth:
            continue
            
        relevantes = set(ground_truth[profile_id]['ofertas_relevantes'])
        
        precision_scores.append(precision_at_k(recommended_ids, relevantes, k))
        recall_scores.append(recall_at_k(recommended_ids, relevantes, k))
        mrr_scores.append(mrr(recommended_ids, relevantes))
        hit_rates.append(hit_rate_at_k(recommended_ids, relevantes, k))
        
    return {
        "Model": model_name,
        "Precision@10": np.mean(precision_scores),
        "Recall@10": np.mean(recall_scores),
        "MRR": np.mean(mrr_scores),
        "Hit Rate@10": np.mean(hit_rates)
    }

def main():
    # Rutas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    profiles_path = os.path.join(data_dir, 'test_profiles.json')
    ground_truth_path = os.path.join(data_dir, 'ground_truth.json')
    
    # Cargar datos
    with open(profiles_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
        
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
        
    # Inicializar modelos
    print("Inicializando modelos...")
    dataset_path = os.path.join(root_dir, 'dataset', 'clean')
    cbf_engine = RecommendationEngine(processed_data_dir=dataset_path)
    baselines = Baselines(processed_data_dir=dataset_path)
    
    results = []
    
    # 1. Evaluar CBF (Propuesto)
    print("\nEvaluando CBF (Propuesto)...")
    cbf_preds = {}
    for profile in profiles:
        recs = cbf_engine.recomendar(profile['texto'], k=10)
        cbf_preds[profile['id']] = [r['id'] for r in recs]
    results.append(evaluate_model("CBF (Propuesto)", cbf_preds, ground_truth))
    
    # 2. Evaluar TF-IDF
    print("Evaluando TF-IDF...")
    tfidf_preds = {}
    for profile in profiles:
        recs = baselines.tfidf_recommendation(profile['texto'], k=10)
        tfidf_preds[profile['id']] = [r['_global_index'] for r in recs]
    results.append(evaluate_model("TF-IDF", tfidf_preds, ground_truth))
    
    # 3. Evaluar Popularidad
    print("Evaluando Popularidad...")
    pop_recs = baselines.popularity_recommendation(k=10)
    pop_ids = [r['_global_index'] for r in pop_recs]
    pop_preds = {p['id']: pop_ids for p in profiles} # Mismo para todos
    results.append(evaluate_model("Popularidad", pop_preds, ground_truth))
    
    # 4. Evaluar Aleatorio
    print("Evaluando Aleatorio...")
    random_preds = {}
    for profile in profiles:
        recs = baselines.random_recommendation(k=10)
        random_preds[profile['id']] = [r['_global_index'] for r in recs]
    results.append(evaluate_model("Aleatorio", random_preds, ground_truth))
    
    # Imprimir Tabla
    df_results = pd.DataFrame(results)
    print("TABLA COMPARATIVA DE RESULTADOS")
    print(df_results.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

if __name__ == "__main__":
    main()
