import argparse
import pickle

import char_rnn
from char_rnn import CharRnn, CharRnnTrainer
char_rnn.device_to_use = 'cpu'

parser = argparse.ArgumentParser(
    description="Create a CharRnn model (along with a trainer).")

parser.add_argument('--dataset', help="A text file to use as the dataset")
parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--chunk_size', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--model_name', help="The path to store the model at")

args = parser.parse_args()

document = open(args.dataset, errors='replace').read()
document_alphabet = ''.join(set(document))

document_model = CharRnn(
    document_alphabet, hidden_size=args.hidden_size, num_layers=args.num_layers)

document_trainer = CharRnnTrainer(
    document_model, document, chunk_size=args.chunk_size,
    batch_size=args.batch_size)

output_file = open(args.model_name, mode='wb')
pickle.dump(document_trainer, output_file)