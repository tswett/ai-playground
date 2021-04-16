import argparse
import pickle

import char_rnn
from char_rnn import CharRnn, CharRnnTrainer
char_rnn.device_to_use = 'cpu'

parser = argparse.ArgumentParser(
    description="Create a CharRnn model (along with a trainer).")

parser.add_argument('--dataset', help="A text file to use as the dataset")
parser.add_argument('--char_encoding', default='utf-8', help="The character encoding to use when reading from the file")
parser.add_argument('--char_errors', default='replace', help="The error behavior when encountering unrecognizable characters in the file")
parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--chunk_size', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--model_name', help="The path to store the model at")

args = parser.parse_args()

print("Reading document...")
document = open(args.dataset, encoding=args.char_encoding, errors=args.char_errors).read()
document_alphabet = ''.join(set(document))

print("Creating model...")
document_model = CharRnn(
    document_alphabet, hidden_size=args.hidden_size, num_layers=args.num_layers)

print("Creating trainer...")
document_trainer = CharRnnTrainer(
    document_model, document, chunk_size=args.chunk_size,
    batch_size=args.batch_size)

print("Writing trainer to file...")
output_file = open(args.model_name, mode='wb')
pickle.dump(document_trainer, output_file)

print("Done.")
