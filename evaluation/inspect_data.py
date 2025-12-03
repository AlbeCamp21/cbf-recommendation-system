import sys
import os

# Agregar la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PLN.searcher import JobSearcher

def inspect_metadata():
    # Apuntar al directorio correcto del dataset
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'clean')
    
    searcher = JobSearcher(processed_data_dir=dataset_dir)
    
    if searcher.job_metadata:
        print("\nClaves en la metadata de la primera oferta:")
        print(searcher.job_metadata[0].keys())
        print("\nValores de muestra:")
        for k, v in searcher.job_metadata[0].items():
            print(f"{k}: {v}")
    else:
        print("No se encontró metadata.")

if __name__ == "__main__":
    inspect_metadata()
