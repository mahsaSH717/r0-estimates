import os
import random

import numpy as np
import torch
from datasets import DatasetDict
from tokenizers import AddedToken
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EarlyStoppingCallback
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.optimization import Adafactor, AdafactorSchedule

from src.util.evaluation_metrics import Metrics

seed = 42
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

path_prefix = "../../../.."
model_is_from_train_set = 'squad_v2_1'
model_id = "google/flan-t5-large"
max_target_length = 512
max_source_length = 512
label_pad_token_id = -100
max_new_tokens = 512
metric_name = "partial_f1s_overall"
dataset_path = f"{path_prefix}/data/processed/final_datasets/{model_is_from_train_set}/text_based/orkg_brp_dataset"

dataset = DatasetDict.load_from_disk(dataset_path)
dataset = dataset.shuffle(seed=seed)
train_dataset = dataset["train"]
eval_dataset = dataset["dev"]
dataset_columns_to_remove = dataset["train"].column_names
dataset_columns_to_remove.remove("template_name")
dataset_columns_to_remove.remove("template_number")

fine_tuned_model_repository = f"{path_prefix}/models/text_based/{model_is_from_train_set}"
tokenizer_repository = f"{path_prefix}/models/text_based/{model_is_from_train_set}/tokenizer"
if not os.path.exists(fine_tuned_model_repository):
    os.makedirs(fine_tuned_model_repository)
    os.makedirs(tokenizer_repository)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_tokens(AddedToken("\n", normalized=False))
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)


def tokenize_function(sample):
    # tokenize inputs
    model_inputs = tokenizer(sample["prompt"], max_length=max_source_length, padding="max_length", truncation=True,
                             return_tensors="pt")

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["response"], max_length=max_target_length, padding="max_length",
                       truncation=True, return_tensors="pt")

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]  # type: ignore
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds):

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)  # type: ignore
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)  # type: ignore
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = Metrics.evaluate_property_wise_text_based(label_list=decoded_labels, prediction_list=decoded_preds)
    result.update(Metrics.evaluate_rouge(label_list=decoded_labels, prediction_list=decoded_preds))
    return result


train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=dataset_columns_to_remove)
eval_tokenized_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=dataset_columns_to_remove)
print(f"Keys of tokenized dataset: {list(train_tokenized_dataset.features)}")

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.001)

training_args = Seq2SeqTrainingArguments(
    output_dir=fine_tuned_model_repository,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    num_train_epochs=30,
    generation_max_length=max_new_tokens,
    gradient_accumulation_steps=8,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    greater_is_better=True,
    logging_dir=f"{fine_tuned_model_repository}/logs",
    logging_strategy="steps",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="tensorboard",
    push_to_hub=False,
    seed=seed
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokenized_dataset,
    eval_dataset=eval_tokenized_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),
    callbacks=([early_stopping_callback])

)

trainer.train()
tokenizer.save_pretrained(tokenizer_repository)
best_ckpt_path = trainer.state.best_model_checkpoint
print(f"best epoch: {best_ckpt_path}")
