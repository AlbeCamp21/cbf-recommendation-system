import json
import os

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    input_file = os.path.join(data_dir, 'prediction_results.json')
    output_file = os.path.join(data_dir, 'ground_truth.json')

    if not os.path.exists(input_file):
        print(f"Error: {input_file} no encontrado.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    ground_truth = {}

    for profile_id, data in results.items():
        expected_category = data['categoria_esperada']
        recommendations = data['recomendaciones']
        
        relevant_ids = []
        for rec in recommendations:
            # Verificar si la categoría coincide
            # Los perfiles usan: desarrollador, contador, marketing, asistente, ingeniero
            # Las etiquetas de archivo son: desarrollador, contador, marketing, asistente, ingeniero, programador, vendedor
            
            rec_category = rec.get('category', '').lower()
            
            # Manejar sinónimos o categorías relacionadas si es necesario
            # Por ahora, coincidencia estricta o mapeo simple
            is_relevant = False
            if expected_category in rec_category:
                is_relevant = True
            elif expected_category == 'desarrollador' and 'programador' in rec_category:
                is_relevant = True
            elif expected_category == 'programador' and 'desarrollador' in rec_category:
                is_relevant = True
            
            if is_relevant:
                relevant_ids.append(rec['id'])
        
        ground_truth[profile_id] = {
            "ofertas_relevantes": relevant_ids,
            "total_evaluadas": len(recommendations)
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    print(f"Ground truth generado en {output_file}")

if __name__ == "__main__":
    main()
