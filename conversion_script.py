import os
import json
from pathlib import Path

ROOT_DIR = Path("/Users/vrishfish/Graph-Augmented-Biomedical-Summarization-BioASQ/data/finetune_data")
OUTPUT_DIR = Path("/Users/vrishfish/Graph-Augmented-Biomedical-Summarization-BioASQ/data/converted_json_finetune_text")
OUTPUT_DIR.mkdir(exist_ok=True)

def collect_pairs(lang_dir: Path, language_code=str):
    full_path = lang_dir/"fulltext"
    sum_path=lang_dir/"summaries" 

    examples = []
    for full_file in full_path.glob("*.txt"):
        base_name = full_file.stem #name of file
        summary_file = sum_path/f"{base_name}_sum.txt"
        
        if not summary_file.exists():
            print("No summary file for {base_name}")
            continue
       
        with open(full_file, "r", encoding = "utf-8") as f:
            full_text = f.read().strip()

        with open(summary_file, "r", encoding = "utf-8") as f:
            summary = f.read().strip()

        examples.append({
            "full_text": full_text,
            "summary": summary,
            "language": language_code
        })
    return examples

def main():
    all_folder = [p for p in ROOT_DIR.iterdir() if p.is_dir()]

    for folder in all_folder:
        folder_name = folder.name 
        parts = folder_name.split("_")

        lang_code = parts[-1]
        print("Processing {folder_name} (language:{lang_code})")

        examples= collect_pairs(folder, lang_code)

        out_file = OUTPUT_DIR/f"{folder_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

    print("ALL DATA CONVERTED TO JSON")

if __name__ == "__main__":
    main()