import argparse

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.dataset.tokenize import get_tokenize_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Inference')

    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--text', type=str, required=True, help='Path to the text to analyze.')

    return parser.parse_args()


def main():
    args = parse_args()
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    tokenize_fn = get_tokenize_fn(tokenizer)
    with open(args.text) as file:
        content = file.read()
    with torch.no_grad():
        x = tokenize_fn(content)
        logits = model(**x).logits
        print("Positive" if torch.argmax(logits, dim=1).item() == 1 else "Negative")


if __name__ == '__main__':
    main()
