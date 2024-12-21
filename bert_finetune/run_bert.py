# %%
import pandas as pd
import os
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import evaluate
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import json
from sklearn.model_selection import train_test_split

from TextDataset import TextDataset
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

from matplotlib import pyplot as plt

# %%
def run(num_to_use = 1, max_epoch = 50):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    # %% [markdown]
    # ## Config tokenizer and model

    # %% [markdown]
    # #### Call BertModel

    # %%
    # Create a config with the desired settings
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)

    # Load the model with the custom config
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    new_tokens = ['\n', '(.)', '(..)', '(...)', 'xxx']
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # %% [markdown]
    # #### Call AutoModel

    # %% [markdown]
    # ## Load Data

    # %%
    train_complete = pd.read_csv("../data/train_complete_v4_4400.csv")
    # true = train_complete[train_complete["example_index"].apply(lambda x: len(str(x)) <= 4)]
    true = train_complete[train_complete["original_index"].apply(lambda x: pd.isna(x))]
    synthetic = train_complete[train_complete["original_index"].apply(lambda x: pd.notna(x))]
    train = pd.concat((true, synthetic.groupby('original_index').head(num_to_use)))
    lines_train = train["line"].to_list()
    labels_train = train["label"].to_list()

    test = pd.read_csv("../data/test_complete_v1_86.csv")
    lines_test = test["line"].to_list()
    labels_test = test["label"].to_list()

    # %%
    print(train.shape, test.shape)

    # %%
    train_dataset = TextDataset(lines_train, labels_train, tokenizer)
    test_dataset = TextDataset(lines_test, labels_test, tokenizer)


    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    # %% [markdown]
    # ## Set Up Model

    # %%
    def model_accuracy(model, dataloader, device):
        """Evaluates the model on the given dataloader and returns accuracy"""
        model.eval()  # Set the model to evaluation mode

        correct_predictions = 0
        total_predictions = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():  # Disable gradient computation
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                outputs = model(**batch)
                logits = outputs.logits

                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                labels = batch['labels']

                # Update correct predictions and totals
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                # Collect all predictions and labels for other metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy
        accuracy = correct_predictions / total_predictions

        return accuracy

    # %%
    optimizer = AdamW(model.parameters(), lr=5e-6)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler with warm-up
    # cap_training_steps = 5000
    # num_epochs = cap_training_steps // len(train_dataloader)
    num_epochs = max_epoch
    cap_training_steps = num_epochs * len(train_dataloader)
    print("num_epochs", num_epochs)
    # num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = int(0.1 * cap_training_steps)  # 10% warm-up
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=cap_training_steps
    )

    # Early stopping parameters
    patience = 5  # Number of epochs to wait for improvement
    best_loss = float('inf')
    patience_counter = 0

    model.to(device)

    # %% [markdown]
    # ## Train

    # %%
    progress_bar = tqdm(range(cap_training_steps))

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
            
        epoch_accuracy = model_accuracy(model, test_dataloader, device)
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}, Learning Rate: {lr_scheduler.get_last_lr()[0]:.8f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "models.nosync/tmp.pt")
            best_epoch = epoch
            best_acc = epoch_accuracy
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience and best_loss < 20:
            print("Early stopping triggered.")
            break
    os.rename("models.nosync/tmp.pt", "models.nosync/v4_{0}_{1}epoch_{2}acc_{3}loss.pt".format(train.shape[0], best_epoch, round(best_acc * 100), round(best_loss,2)))
