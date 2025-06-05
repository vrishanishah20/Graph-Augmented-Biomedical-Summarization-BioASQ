import json
from pathlib import Path
from datasets import Dataset
from transformers import MT5Tokenizer
import shutil

# load MT5Tokenizer using spiece.model 
spiece_model_path = "/Users/vrishfish/.cache/huggingface/hub/models--google--mt5-small/snapshots/73fb5dbe4756edadc8fbe8c769b0a109493acf7a/spiece.model"

tokenizer = MT5Tokenizer(
    vocab_file=spiece_model_path,
    extra_ids=112,
    legacy=False
)
tokenizer.save_pretrained("/Users/vrishfish/Graph-Augmented-Biomedical-Summarization-BioASQ/clean_mt5_tokenizer")

print("\n=== DEBUG ===")
print("Tokenizer class:", type(tokenizer))
print("Tokenizer vocab size:", tokenizer.vocab_size)
print("Total tokens (len):", len(tokenizer))
print("Added tokens:", tokenizer.added_tokens_encoder.keys())
print("EOS token:", tokenizer.eos_token, tokenizer.eos_token_id)

expected_len = 250100 + 112  # original vocab + extra tokens
assert len(tokenizer) == expected_len, f"Expected {expected_len}, got {len(tokenizer)}"
print("All good")

# Prompt definitions for multilingual summarization 
PROMPTS = {
    "en": "Summarize the following biomedical clinical report",
    "es": "Resume el siguiente informe clínico:",
    "fr": "Résumez le rapport clinique suivant:",
    "pt": "Resuma o seguinte relatório clínico:"
}

def clean_target(text):
    cleaned = text.replace("</s>", "").strip()
    return cleaned if cleaned else "[EMPTY]"

def load_and_format(lang_code):
    path = f"/Users/vrishfish/Graph-Augmented-Biomedical-Summarization-BioASQ/data/converted_json_finetune_text/multiclinsum_large-scale_train_{lang_code}.json"
    with open(path) as f:
        raw_data = json.load(f)
    return [
        {
            "input": f"{PROMPTS[lang_code]} {example['full_text']}",
            "target": clean_target(example["summary"])
        }
        for example in raw_data
    ]

# Load and format full multilingual dataset 
dataset = Dataset.from_list(
    [ex for lang in ["en", "es", "fr", "pt"] for ex in load_and_format(lang)]
)

def clean_and_add_eos(text):
    # Strip any existing </s> to avoid duplicates
    cleaned = text.replace("</s>", "").strip()
    return cleaned + " </s>"

def tokenize_function(example):
    inputs = example["input"]
    targets = [clean_and_add_eos(t) for t in example["target"]]

    input_encodings = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding=False
    )

    label_encodings = tokenizer(
        text_target=targets,
        max_length=600,
        truncation=True,
        padding=False,
        add_special_tokens=False
    )

    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": label_encodings["input_ids"]
    }

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["input", "target"]
)

# save
save_path = Path("/Users/vrishfish/Graph-Augmented-Biomedical-Summarization-BioASQ/tokenized_large_dataset")
tokenized_dataset.save_to_disk(save_path)
print("\n=== Verification ===")
print(f"Saved to: {save_path}")
print("Tokenizer class:", type(tokenizer))
print("Tokenizer vocab size:", tokenizer.vocab_size)


sample_raw = dataset[0]["target"]
print("\nRaw target sample:", sample_raw)
print("Contains </s>?", "</s>" in sample_raw)

sample_input = tokenizer.decode(tokenized_dataset[0]["input_ids"])
sample_label = tokenizer.decode(tokenized_dataset[0]["labels"], skip_special_tokens=True)
print("\nDecoded input:", sample_input)
print("Decoded label:", sample_label)
print("Label ends with EOS?", tokenized_dataset[0]["labels"][-1] == tokenizer.eos_token_id)

all_labels = [label for sublist in tokenized_dataset["labels"] for label in sublist]
eos_count = sum(1 for label in all_labels if label == tokenizer.eos_token_id)
print(f"\nTotal EOS in labels: {eos_count} (should equal number of samples: {len(tokenized_dataset)})")
