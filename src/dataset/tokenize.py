import torch


def get_tokenize_fn(tokenizer):
    def tokenize_fn(text):
        ids = [tokenizer.encode(sequence,
                                add_special_tokens=True,
                                padding='max_length',
                                truncation=True,
                                max_length=512,  # max input sequence length for BERT-based models
                                return_tensors='pt') for sequence in text]
        return torch.cat(ids, dim=0)

    return tokenize_fn
