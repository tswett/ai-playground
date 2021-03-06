import argparse
import pickle

from ai_playground import char_rnn
from ai_playground.char_rnn import CharRnn, CharRnnTrainer
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

if __name__ == '__main__':
    args = parser.parse_args()

    print("Reading document...")
    with open(args.dataset, encoding=args.char_encoding, errors=args.char_errors) as document_file:
        document = document_file.read()
    document_alphabet = ''.join(set(document))

    print("Creating model...")
    document_model = CharRnn(
        document_alphabet, hidden_size=args.hidden_size, num_layers=args.num_layers)

    print("Creating trainer...")
    document_trainer = CharRnnTrainer(
        document_model,
        document,
        dataset_filename = args.dataset,
        char_encoding = args.char_encoding,
        char_errors = args.char_errors,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size)

    print("Writing trainer to file...")
    with open(args.model_name, mode='wb') as output_file:
        pickle.dump(document_trainer, output_file)

    print("Done.")
