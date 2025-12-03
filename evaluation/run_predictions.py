import json
import os
import sys

# Agregar la raíz del proyecto al path para asegurar que se encuentre el módulo PLN
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'PLN'))

from PLN.recommender import RecommendationEngine

def main():
    # Definir rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    profiles_file = os.path.join(data_dir, 'test_profiles.json')
    output_file = os.path.join(data_dir, 'prediction_results.json')
    dataset_path = os.path.join(project_root, 'dataset', 'clean')

    # Cargar perfiles de prueba
    if not os.path.exists(profiles_file):
        print(f"Error: {profiles_file} no encontrado.")
        return

    with open(profiles_file, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    # Inicializar motor
    print(f"Conectando al dataset en: {dataset_path}")
    
    try:
        engine = RecommendationEngine(processed_data_dir=dataset_path)
    except Exception as e:
        print(f"Error inicializando el motor: {e}")
        return

    results_map = {}

    for profile in profiles:
        print(f"Procesando perfil: {profile['id']} ({profile['categoria_esperada']})")
        try:
            recommendations = engine.recomendar(profile['texto'], k=20, verbose=True)
            
            results_map[profile['id']] = {
                "categoria_esperada": profile['categoria_esperada'],
                "texto_perfil": profile['texto'],
                "recomendaciones": recommendations
            }
        except Exception as e:
            print(f"Error procesando perfil {profile['id']}: {e}")

    # Guardar resultados
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_map, f, indent=2, ensure_ascii=False)
    
    print(f"Resultados guardados en {output_file}")

if __name__ == "__main__":
    main()
