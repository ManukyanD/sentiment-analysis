def get_tokenize_fn(tokenizer):
    def tokenize_fn(text):
        return tokenizer(text,
                         return_tensors="pt",
                         padding=True,
                         truncation=True,
                         max_length=512,
                         add_special_tokens=True)

    return tokenize_fn
