import os
import pandas as pd
import numpy as np
import sys
import csv
import nltk
import torch
from torch import nn
from tqdm.auto import tqdm
import evaluate
from transformers import TrainingArguments, Trainer, AutoModel, DataCollatorWithPadding, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")

# cuda setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# model setting
base_model = "./deberta-v3-large"
config = AutoConfig.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
save_path = './results'

# load the data
df_train = pd.read_json("./data/train_set.json")
df_test = pd.read_json("./data/test_set.json")

# parameters
n_fold = 5
batch_size = 8
EPOCHS = 1
lr = 1e-5

def preprocess_function(examples):
    return tokenizer(examples["text"], padding = 'max_length', max_length = 256, truncation = True)

def compute_metrics(pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    x, label = pred
    prediction = np.argmax(x, axis = -1)
    accuracy = accuracy_metric.compute(prediction = prediction, references=label)["accuracy"]
    f1 = f1_metric.compute(prediction =prediction, references = label)["f1"]
    return {"accuracy": accuracy, "f1": f1}

class FeedbackModel(nn.Module):
    def __init__(self, num_labels = 2):
        super(FeedbackModel, self).__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(base_model)
        self.model = AutoModel.from_pretrained(base_model)
        self.hidden_size = self.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids = None, attention_mask = None, labels = None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = self.model(input_ids = input_ids, attention_mask = attention_mask)

        hidden_state = output[0]
        pooled_output = hidden_state[:, 0]  
        x = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(x.view(-1, self.num_labels), labels.view(-1))
            return TokenClassifierOutput(loss = loss, logits = x, hidden_states = output.hidden_states, attentions = output.attentions)
        else:
            return x


# k-fold
n_splits = n_fold
k_fold = StratifiedKFold(n_splits = n_splits, random_state = 42, shuffle = True)


# start training
print("Start training on the train dataset")

model_metrics = {'eval_loss': 0, 'eval_accuracy': 0, 'eval_f1': 0}
for idx, (train_idxs, val_idxs) in enumerate(k_fold.split(df_train["text"], df_train["label"])):
    print(f"\nFold {idx}\n")
    model = FeedbackModel(2)
    model.to(device)
    df_train_fold = df_train.iloc[train_idxs]
    df_val_fold = df_train.iloc[val_idxs]
    train_dataset = Dataset.from_pandas(df_train_fold)
    val_dataset = Dataset.from_pandas(df_val_fold)
    train = train_dataset.map(preprocess_function, batched = True)
    val = val_dataset.map(preprocess_function, batched = True)

    training_args = TrainingArguments(
        output_dir = save_path,
        learning_rate = lr,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs = EPOCHS,
        weight_decay = 0.01,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        push_to_hub = False,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train,
        eval_dataset = val,
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics,
    )

    print(f"Training fold {idx}")
    trainer.train()

    print(f"\nEvaluating fold {idx}\n")
    metrics = trainer.evaluate()
    model_metrics["eval_loss"] += metrics["eval_loss"]
    model_metrics["eval_accuracy"] += metrics["eval_accuracy"]
    model_metrics["eval_f1"] += metrics["eval_f1"]
    print(f"Fold {idx} metrics: {metrics}")
    break

for k, v in model_metrics.items():
    model_metrics[k] = v / n_splits

print(f"Global metrics: {model_metrics}")
print(f"Saving the model to ")
trainer.save_model(save_path)


# predict
test_dataset = Dataset.from_pandas(df_test)
test = test_dataset.map(preprocess_function, batched = True)
test_dataloader = DataLoader(test, batch_size = 1)

predictions = []


# start predicting
print("Start predicting on the test dataset")

model.eval()
for batch in test_dataloader:
    input_ids = torch.stack(batch["input_ids"]).view((1, -1))
    attention_mask = torch.stack(batch["attention_mask"]).view((1, -1))
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}

    with torch.no_grad():
        output = model.forward(**batch)

    prediction = torch.argmax(output, dim=-1).item()
    predictions.append(prediction)


# writing the results
print("Start writing the submission.csv")
with open("./data/submission.csv", "w") as f:
    csv_out = csv.writer(f)
    csv_out.writerow(['id', 'label'])

    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])
