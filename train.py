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
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        accurate_predictions = 0
        model.train()
        for index, batch in enumerate(train_loader):
            model.zero_grad()
            x, y = batch
            x = to_device(tokenize_fn(x))
            y = to_device(y)
            loss, logits = model(x, attention_mask=(x > 0), labels=y, return_dict=False)
            total_loss += loss.item()
            accurate_predictions += (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).sum().item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            if (index + 1) % 10 == 0:
                print(
                    f'Batches: up to {index + 1}, '
                    f'Loss: {total_loss / (index + 1)}, '
                    f'Accuracy: {round(accurate_predictions / (args.batch_size * (index + 1)) * 100, 2)} %')
        report('Training', total_loss / len(train_loader), accurate_predictions / len(train_loader) / args.batch_size,
               epoch)
        evaluate(epoch, model, test_loader, tokenize_fn, args.batch_size)
        checkpoint(epoch, model, args.checkpoints_dir)


def evaluate(epoch, model, test_loader, tokenize_fn, batch_size):
    model.eval()
    test_loss = 0
    accurate_predictions = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = to_device(tokenize_fn(x))
            y = to_device(y)
            loss, logits = model(x, attention_mask=(x > 0), labels=y, return_dict=False)
            test_loss += loss.item()
            accurate_predictions += (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).sum().item()
    report('Testing', test_loss / len(test_loader), accurate_predictions / len(test_loader) / batch_size, epoch)


def report(phase, loss, accuracy, epoch):
    summary_writer.add_scalar(f'{phase} loss', loss, epoch)
    print(f'Phase: {phase}, Epoch: {epoch}, Loss: {loss}, Accuracy: {round(accuracy * 100, 2)} %')


def checkpoint(epoch_num, model, checkpoint_dir):
    torch.save(model, os.path.join(checkpoint_dir, f'epoch-{epoch_num}.pt'))


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


if __name__ == '__main__':
    main()
