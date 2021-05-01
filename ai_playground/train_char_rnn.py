import argparse
import datetime
import itertools
import pickle

from ai_playground import char_rnn
char_rnn.device_to_use = 'cpu'

parser = argparse.ArgumentParser(
    description="Train and sample from a CharRnnTrainer.")

parser.add_argument('--dataset', help="A text file to use as the dataset")
parser.add_argument('--char_encoding', help="The character encoding to use when reading from the file")
parser.add_argument('--char_errors', help="The error behavior when encountering unrecognizable characters in the file")
parser.add_argument('--model_name', help="The path to retrieve the model from")
parser.add_argument('--save_model_name', help="The path to save new models at")
parser.add_argument('--save_every', type=int, help="Save the model every this-many training steps")
parser.add_argument('--sample_chars', type=int, default=20, help="Sample this many characters in each training step")

if __name__ == '__main__':
    args = parser.parse_args()

    print("Loading model...")
    with open(args.model_name, mode='rb') as trainer_file:
        trainer = pickle.load(trainer_file)

    print("Loading training dataset...")
    trainer.load_corpus(args.dataset, args.char_encoding, args.char_errors)

    sampler = trainer.model.sample_randomly('. ', temperature=0.8)

    print("Starting training.")
    for i in itertools.count(start=1):
        loss = trainer.step()

        prefix = f'{i} {loss:.4f}'
        for char in itertools.islice(sampler, args.sample_chars):
            sample = char.replace('\n', f'\n{prefix} ')
            print(sample, end='', flush=True)
        
        if args.save_every is not None and i % args.save_every == 0:
            date_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            save_name = f'{args.save_model_name}-{date_string}'
            save_file = open(save_name, mode='wb')
            pickle.dump(trainer, save_file)
            save_file.close()
