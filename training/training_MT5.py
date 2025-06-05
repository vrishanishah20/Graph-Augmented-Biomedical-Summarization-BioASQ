import json 
from pathlib import Path
from datasets import load_from_disk, Dataset
import transformers
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch.nn.functional as F
import evaluate
import re
import torch
import numpy as np
import torch.utils.checkpoint
torch.utils.checkpoint._use_reentrant = False

# Configurations
DATASET_PATH = "/content/drive/MyDrive/Biomedical-Summarization-Using-GraphRAG/tokenized_large_dataset"
MODEL_NAME = "google/mt5-small"
OUT_DIR = Path("/content/drive/MyDrive/Biomedical-Summarization-Using-GraphRAG/mt5-biomed-checkpoints")
LOGGING_DIR = Path("/content/drive/MyDrive/Biomedical-Summarization-Using-GraphRAG/mt5_logs")
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 3e-5

# Initialize directories
OUT_DIR.mkdir(exist_ok=True, parents=True)
LOGGING_DIR.mkdir(exist_ok=True, parents=True)

#Custom generation model for better rouge scores
# class MT5WithRepeatPenalty(MT5ForConditionalGeneration):
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         labels=None,
#         **kwargs
#     ):
#         # Run the base model
#         outputs = super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask,
#             labels=labels,
#             **kwargs
#         )

#         loss = outputs.loss

#         # Only apply trigram penalty during training (not generation)
#         if labels is not None:
#             penalty_weight = 0.3
#             repeat_counts = []

#             for label_seq in labels:
#                 label_seq = label_seq[label_seq != -100]  # remove padding
#                 tokens = label_seq.tolist()
#                 seen = set()
#                 repeat_count = 0
#                 for i in range(len(tokens) - 2):
#                     trigram = tuple(tokens[i:i+3])
#                     if trigram in seen:
#                         repeat_count += 1
#                     else:
#                         seen.add(trigram)
#                 repeat_counts.append(repeat_count)

#             repeat_tensor = torch.tensor(repeat_counts, dtype=torch.float, device=loss.device)
#             penalty = penalty_weight * repeat_tensor.mean()
#             loss = loss + penalty

#         return transformers.modeling_outputs.Seq2SeqLMOutput(
#             loss=loss,
#             logits=outputs.logits,
#             past_key_values=outputs.past_key_values,
#             decoder_hidden_states=outputs.decoder_hidden_states,
#             decoder_attentions=outputs.decoder_attentions,
#             cross_attentions=outputs.cross_attentions,
#             encoder_last_hidden_state=outputs.encoder_last_hidden_state,
#             encoder_hidden_states=outputs.encoder_hidden_states,
#             encoder_attentions=outputs.encoder_attentions,
#         )


# Loading model and tokenizer
print("Loading model and tokenizer")
tokenizer = MT5Tokenizer.from_pretrained("/content/drive/MyDrive/Biomedical-Summarization-Using-GraphRAG/clean_mt5_tokenizer", legacy=False)
model = MT5ForConditionalGeneration.from_pretrained(
  "google/mt5-small"
  # force_download=True, 
  # cache_dir="/tmp/mt5_cache" #cleaning cache
)

if model.config.vocab_size != len(tokenizer):
    print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}") #to match the mt5 tokenizer vocab.size to model vocab.size
    model.resize_token_embeddings(len(tokenizer))

print(f"Tokenizer usable vocab size: {len(tokenizer)}")
print(f"Model vocab size: {model.config.vocab_size}")
assert len(tokenizer) == model.config.vocab_size, f"Tokenizer and model vocab mismatch: {len(tokenizer)} vs {model.config.vocab_size}"


# Loading and splitting dataset
print("Loading dataset")
dataset = load_from_disk(DATASET_PATH)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,  
    pad_to_multiple_of=8,
    label_pad_token_id=-100
)


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir=LOGGING_DIR,
    logging_steps=50,
    report_to="tensorboard",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    generation_max_length=256,
    generation_num_beams=4,
    fp16=True,  
    optim="adamw_torch",
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
    dataloader_num_workers=2,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    # fp16_full_eval=True,
    label_smoothing_factor=0.1,
    fp16_opt_level = 'O2'

)
def clean_prediction(pred):
    import re
    pred = re.sub(r"<extra_id_\d+>", "", pred) #removing extra ids for better rouge score
    pred = re.sub(r"\b(\w+ \w+)\s+\1\b", r"\1", pred)  # Repeated bigrams
    pred = pred.strip()
    return pred

# Metrics
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Unpack from Seq2SeqTrainer)
    if isinstance(preds, tuple):
        preds = preds[0]

    # Convert to numpy array
    preds = np.array(preds)
    labels = np.array(labels)

    # PREDICTIONS HANDLING
    # Clamp predictions to valid token ID range
    preds = np.clip(preds, 0, len(tokenizer) - 1)

    # Check for remaining invalid tokens
    invalid_mask = (preds < 0) | (preds >= len(tokenizer))
    if np.any(invalid_mask):
        print(f"Warning: {np.sum(invalid_mask)} invalid token IDs after clamping")

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    #LABELS HANDLING
    # Replace -100 with pad_token_id before decoding
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    labels = np.clip(labels, 0, len(tokenizer) - 1)  # Extra safety

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    #EMPTY STRING HANDLING 
    decoded_preds = [p.strip() if p.strip() else "[EMPTY]" for p in decoded_preds]
    decoded_labels = [l.strip() if l.strip() else "[EMPTY]" for l in decoded_labels]
    decoded_preds = [clean_prediction(p) for p in decoded_preds]
    decoded_labels = [clean_prediction(l) for l in decoded_labels]

    # metrics
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True
    )

    result["prediction_lengths"] = np.mean([len(p) for p in decoded_preds])
    result["label_lengths"] = np.mean([len(l) for l in decoded_labels])

    return {k: round(v, 4) for k, v in result.items()}


# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# # Overfit test
# print("\n=== Starting overfit test ===")
# tiny_train = train_dataset.select(range(10))
# tiny_val = val_dataset.select(range(2))

# # Special args for overfit test
# tiny_args = Seq2SeqTrainingArguments(
#     output_dir=OUT_DIR / "tiny",
#     num_train_epochs=10,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=2,
#     learning_rate=1e-5,
#     fp16=False,
#     optim="adamw_torch",
#     weight_decay=0.0,
#     gradient_accumulation_steps=1,
#     report_to="none",
#     save_strategy="no",
#     evaluation_strategy="steps",
#     eval_steps=50,
#     logging_steps=10,
#     predict_with_generate=True,
#     remove_unused_columns=False
# )

# cloning model for overfit test
# tiny_model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
# tiny_trainer = Seq2SeqTrainer(
#     model=tiny_model,
#     args=tiny_args,
#     train_dataset=tiny_train,
#     eval_dataset=tiny_val,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics
# )

# Run overfit test
# print("Running overfit test...")
# tiny_trainer.train()
# print("Overfit test complete!")

# Full training
print("\nStarting main training")
train_result = trainer.train()
trainer.save_model(OUT_DIR / "final_model")
tokenizer.save_pretrained(OUT_DIR / "final_model")

# Evaluation
print("\nFinal evaluation")
metrics = trainer.evaluate()
print("Validation metrics:", metrics)

# Generate predictions
print("\nGenerating predictions...")
predictions = trainer.predict(val_dataset)
decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(
    [[l for l in label if l != -100] for label in predictions.label_ids],
    skip_special_tokens=True
)

# Save predictions
with open(OUT_DIR / "predictions.txt", "w") as f:
    for pred, label in zip(decoded_preds, decoded_labels):
        f.write(f"PRED: {pred}\nLABEL: {label}\n\n")