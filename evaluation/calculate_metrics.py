import json
import os

def precision_at_k(recomendados: list, relevantes: set, k: int) -> float:
    """
    Precision@k = (ofertas relevantes en top-k) / k
    """
    top_k = recomendados[:k]
    relevantes_en_top_k = len([x for x in top_k if x in relevantes])
    return relevantes_en_top_k / k

def recall_at_k(recomendados: list, relevantes: set, k: int) -> float:
    """
    Recall@k = (relevantes en top-k) / (total de relevantes)
    """
    top_k = recomendados[:k]
    relevantes_en_top_k = len([x for x in top_k if x in relevantes])
    return relevantes_en_top_k / len(relevantes) if relevantes else 0

def mrr(recomendados: list, relevantes: set) -> float:
    """
    MRR (Mean Reciprocal Rank) = 1 / posición del primer relevante
    """
    for i, item in enumerate(recomendados, 1):
        if item in relevantes:
            return 1.0 / i
    return 0.0

def hit_rate_at_k(recomendados: list, relevantes: set, k: int) -> float:
    """
    Hit Rate@k = 1 si hay al menos 1 relevante en top-k, 0 si no
    """
    top_k = recomendados[:k]
    return 1.0 if any(x in relevantes for x in top_k) else 0.0

def evaluar_sistema():
    # Cargar resultados
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    results_file = os.path.join(data_dir, 'prediction_results.json')
    ground_truth_file = os.path.join(data_dir, 'ground_truth.json')

    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    hit_rates = []
    
    for profile_id, data in results_data.items():
        # Obtener IDs de recomendaciones
        ids_recomendados = [r['id'] for r in data['recomendaciones']]
        
        # Obtener IDs relevantes
        if profile_id not in ground_truth:
            print(f"Advertencia: No hay ground truth para {profile_id}")
            continue
            
        relevantes = set(ground_truth[profile_id]['ofertas_relevantes'])
        
        # Calcular métricas
        precision_scores.append(precision_at_k(ids_recomendados, relevantes, 10))
        recall_scores.append(recall_at_k(ids_recomendados, relevantes, 10))
        mrr_scores.append(mrr(ids_recomendados, relevantes))
        hit_rates.append(hit_rate_at_k(ids_recomendados, relevantes, 10))

    if precision_scores:
        print(f"Precision@10:  {sum(precision_scores)/len(precision_scores):.4f}")
        print(f"Recall@10:     {sum(recall_scores)/len(recall_scores):.4f}")
        print(f"MRR:           {sum(mrr_scores)/len(mrr_scores):.4f}")
        print(f"Hit Rate@10:   {sum(hit_rates)/len(hit_rates):.4f}")
    else:
        print("No hay resultados para evaluar.")

if __name__ == "__main__":
    evaluar_sistema()
