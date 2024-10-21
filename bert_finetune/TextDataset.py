import torch

class TextDataset():
    def __init__(self, lines, labels, tokenizer, max_length=512):
        self.lines = lines
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            line,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )

        # Return input_ids and attention_mask as tensors
        input_ids = encoding['input_ids'].squeeze()      # shape: [max_length]
        attention_mask = encoding['attention_mask'].squeeze()  # shape: [max_length]
        token_type_ids = encoding['token_type_ids'].squeeze()
        label = torch.tensor(label, dtype=torch.long)

        return {
            'labels': label,
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }