def get_tokenize_fn(tokenizer):
    def tokenize_fn(text):
        return tokenizer(text,
                         return_tensors="pt",
                         padding='max_length',
                         truncation=True,
                         max_length=512,  # max input sequence length for BERT-based models
                         add_special_tokens=True)

    return tokenize_fn
