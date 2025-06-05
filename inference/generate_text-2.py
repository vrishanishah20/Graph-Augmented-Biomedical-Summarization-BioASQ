import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, DataCollatorForSeq2Seq
from datasets import load_from_disk
from pathlib import Path
import os

MODEL_DIR = "/content/drive/MyDrive/Biomedical-Summarization-Using-GraphRAG/mt5-biomed-checkpoints/final_model"
DATASET_PATH = "/content/drive/MyDrive/Biomedical-Summarization-Using-GraphRAG/tokenized_dataset"
OUTPUT_PATH = Path(MODEL_DIR) / "generated_predictions.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
MAX_LENGTH = 200  # faster than 256
NUM_BEAMS = 2     # reduced from 4 for speed and memeroy

# loading model and tokenizer
print("Loading tokenizer and model...")
tokenizer = MT5Tokenizer.from_pretrained(MODEL_DIR)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)

print("Loading tokenized dataset...")
dataset = load_from_disk(DATASET_PATH)
test_dataset = dataset["test"] if "test" in dataset else dataset

#will form a batch by using a list of dataset elements as input
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

#generation arguments
gen_kwargs = {
    "max_length": MAX_LENGTH,
    "num_beams": NUM_BEAMS,
    "do_sample": False
}

# ##Use only 15 examples 
# subset = test_dataset.select(range(15))

print("Generating predictions...")
with open(OUTPUT_PATH, "w") as f:
    for i in range(0, len(subset), BATCH_SIZE):
        batch = test_dataset[i: i + BATCH_SIZE]
        features = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        # features = [x for x in batch]
        batch_collated = data_collator(features)

        input_ids = batch_collated["input_ids"].to(DEVICE)
        attention_mask = batch_collated["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, pred in enumerate(decoded):
            f.write(f"{i + j + 1}: {pred.strip()}\n")

        print(f"✓ Batch {i // BATCH_SIZE + 1} done")

print(f"Saved predictions for 15 examples to: {OUTPUT_PATH}")
