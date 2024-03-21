import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModelForSequenceClassification

from src.dataset.imdb_dataset import IMDBDataset
from src.dataset.tokenize import get_tokenize_fn
from src.util.arg_parser import parse_args
from src.util.device import to_device

summary_writer = SummaryWriter()


def fit(args, model, train_loader, test_loader, tokenize_fn, optimizer, scheduler):
    running_loss = 0
    accurate_predictions = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for index, batch in enumerate(train_loader):
            model.zero_grad()
            x, y = batch
            x = to_device(tokenize_fn(x))
            y = to_device(y)
            loss, logits = model(**x, labels=y, return_dict=False)
            running_loss += loss.item()
            accurate_predictions += (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).sum().item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step = (epoch - 1) * len(train_loader) + index + 1
            if step % 10 == 0:
                avg_loss = running_loss / 10
                avg_accuracy = accurate_predictions / (10 * args.batch_size)
                report(avg_loss, avg_accuracy, step)
                checkpoint(epoch, model, args.checkpoints_dir)
                running_loss = 0
                accurate_predictions = 0
        evaluate(epoch, model, test_loader, tokenize_fn, args.batch_size)


def evaluate(epoch, model, test_loader, tokenize_fn, batch_size):
    model.eval()
    total_loss = 0
    accurate_predictions = 0
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            x, y = batch
            x = to_device(tokenize_fn(x))
            y = to_device(y)
            loss, logits = model(**x, labels=y, return_dict=False)
            total_loss += loss.item()
            accurate_predictions += (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).sum().item()
            batch_num = index + 1
            if batch_num % 10 == 0:
                print(
                    f'Epoch: {epoch}, '
                    f'Test batch: up to {batch_num}, '
                    f'Loss: {total_loss / batch_num}, '
                    f'Accuracy: {round(accurate_predictions / (batch_num * batch_size) * 100, 2)} %')


def report(loss, accuracy, step):
    summary_writer.add_scalar(f'Train/Loss', loss, step)
    summary_writer.add_scalar(f'Train/Accuracy', accuracy, step)
    print(f'Step: {step}, Loss: {loss}, Accuracy: {round(accuracy * 100, 2)} %')


def checkpoint(epoch_num, model, checkpoint_dir):
    model.save_pretrained(os.path.join(checkpoint_dir, f'epoch-{epoch_num}'))


def main():
    args = parse_args()
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    tokenize_fn = get_tokenize_fn(tokenizer)

    train_loader = DataLoader(IMDBDataset('train'), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(IMDBDataset('test'), batch_size=args.batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2)
    to_device(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * args.epochs)

    fit(args, model, train_loader, test_loader, tokenize_fn, optimizer, scheduler)
    summary_writer.close()


if __name__ == '__main__':
    main()
