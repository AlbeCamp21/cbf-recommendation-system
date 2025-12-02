import argparse
import json
import pandas as pd
import re
import os
import pickle
from sentence_transformers import SentenceTransformer

def parse_cli_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Embedding Generator")
    
    parser.add_argument(
        "input_path", 
        type=str, 
        help="Path to the input JSON file (e.g., data/avisos_programador.json)"
    )

    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Path for the output .pkl file. If empty, generates name automatically."
    )

    return parser.parse_args()

def generate_output_path(input_path):
    """
    Generates an output filename based on the input filename.
    Input:  'dataset/avisos_programador.json'
    Output: 'dataset/processed/vectors_programador.pkl'
    """
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)

    name_without_ext = os.path.splitext(filename)[0]    # "avisos_programador"
    job_name = name_without_ext.replace("avisos_", "")  # "programador"

    new_filename = f"vectors_{job_name}.pkl"

    return os.path.join(directory, "processed", new_filename)

def clean_text_content(text_input):
    """
    Normalizes text: converts to lowercase, removes newlines, tabs, 
    and special characters.
    """
    if not isinstance(text_input, str):
        return ""
    
    normalized_text = text_input.lower()
    normalized_text = re.sub(r'[\n\t\r]', ' ', normalized_text)
    normalized_text = re.sub(r'[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ\s]', '', normalized_text)
    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
    return normalized_text

def main():
    args = parse_cli_arguments()

    if args.output:
        output_path = args.output
    else:
        output_path = generate_output_path(args.input_path)

    if not os.path.exists(args.input_path):
        print(f" ERROR: Input file not found at: {args.input_path}")
        return

    print(f"Loading raw data from: {args.input_path}")
    print(f"Target output file: {output_path}")

    try:
        with open(args.input_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
    except Exception as e:
        print(f"ERROR reading JSON: {e}")
        return

    jobs_df = pd.DataFrame(raw_data)
    
    if jobs_df.empty:
        print("WARNING: The JSON file is empty.")
        return

    print(f"Loaded {len(jobs_df)} job offers.")

    print("Preprocessing text...")
    jobs_df['combined_text'] = jobs_df['title'].fillna('') + ". " + jobs_df['description'].fillna('')
    jobs_df['cleaned_text'] = jobs_df['combined_text'].apply(clean_text_content)

    print("Loading AI Model...")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("Generating vectors (Embeddings)...")
    job_embeddings = embedding_model.encode(jobs_df['cleaned_text'].tolist(), show_progress_bar=True)

    final_payload = {
        "metadata": jobs_df.to_dict(orient='records'),
        "embeddings": job_embeddings
    }

    print(f"Saving processed data...")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating missing directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)


    with open(output_path, "wb") as file:
        pickle.dump(final_payload, file)

    print("\nCOMPLETED SUCCESSFULLY.")
    print(f"   Generated file: {output_path}")

if __name__ == "__main__":
    main()