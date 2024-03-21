import argparse
import os.path


def parse_args():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Training')

    parser.add_argument('--checkpoints-dir', type=str, default=os.path.join('.', 'checkpoints'),
                        help='The directory to save checkpoints to (default: "./checkpoints").')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs (default: 3).')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Base learning rate (default: 2e-5).')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01).')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16).')

    return parser.parse_args()
