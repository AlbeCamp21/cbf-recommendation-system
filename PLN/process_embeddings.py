import os
import re
import json
import pickle
import argparse
import warnings
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

class JobOfferProcessor:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        print(f"Inicializando procesador...")
        self.model = SentenceTransformer(model_name)
        print("OK - Modelo IA cargado.")

    def clean_text(self, text: str) -> str:
        """Limpieza profunda para que la IA entienda mejor."""
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[\n\t\r]', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def load_and_tag_from_folder(self, folder_path: str) -> pd.DataFrame:
        """
        1. Lee los archivos JSON.
        2. Extrae la categoría del nombre del archivo.
        3. Etiqueta cada oferta con esa categoría.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"No existe: {folder_path}")
            
        all_records = []
        files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        print(f"Procesando {len(files)} archivos para extracción de categorías...")

        for filename in tqdm(files):
            file_tag = filename.replace("avisos_", "").replace(".json", "")
            
            full_path = os.path.join(folder_path, filename)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        for offer in data:
                            offer['category'] = file_tag
                        
                        all_records.extend(data)
            except Exception as e:
                print(f"X - Error en {filename}: {e}")

        return pd.DataFrame(all_records)

    def filter_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina duplicados comparando SOLO LA DESCRIPCIÓN.
        """
        initial_count = len(df)
        
        # 1. Eliminar vacíos
        df = df.dropna(subset=['title', 'description'])
        df = df[df['description'].str.strip() != ""]
        
        # 2. ELIMINAR DUPLICADOS POR DESCRIPCIÓN
        df = df.drop_duplicates(subset=['title', 'description'], keep='first')
        
        # Resetear índice para que quede ordenado
        df = df.reset_index(drop=True)
        
        removed = initial_count - len(df)
        print(f"Limpieza (Duplicados por Descripción):")
        print(f"{initial_count} iniciales -> {len(df)} únicos ({removed} eliminados).")
        
        return df

    def run_pipeline(self, input_folder: str, output_path: str):
        folder_abs = os.path.abspath(input_folder)
        output_abs = os.path.abspath(output_path)
        
        print(f"\n--- INICIO DEL PROCESO ---")
        
        # 1. Carga y Etiquetado
        df = self.load_and_tag_from_folder(folder_abs)
        if df.empty:
            print("No hay datos.")
            return

        # Mostrar cuántos hay por categoría antes de limpiar
        print("\nConteo por categoría detectada (Antes de limpiar):")
        print(df['category'].value_counts())

        # 2. Limpieza (Duplicados por Descripción)
        df = self.filter_and_deduplicate(df)
        if df.empty: return

        # 3. NLP Cleaning (Para la IA)
        print("\nGenerando texto limpio para la IA...")
        combined = df['title'].fillna('') + " " + df['category'].fillna('') + ". " + df['description'].fillna('')
        df['cleaned_text'] = combined.apply(self.clean_text)
        df = df[df['cleaned_text'] != ""]

        # 4. Vectorización
        print(f"Creando Embeddings para {len(df)} ofertas...")
        embeddings = self.model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
        
        # 5. Guardado
        payload = {
            "metadata": df.to_dict(orient='records'),
            "embeddings": embeddings
        }
        
        os.makedirs(os.path.dirname(output_abs), exist_ok=True)
        with open(output_abs, "wb") as f:
            pickle.dump(payload, f)
        
        print(f"\nOK - GUARDADO EXITOSO: {output_abs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Carpeta con los JSON")
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    final_out = args.output if args.output else os.path.join(args.input_folder, "processed", "vectors_dataset_final.pkl")
    
    JobOfferProcessor().run_pipeline(args.input_folder, final_out)