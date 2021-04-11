import itertools
import timeit

from src import char_rnn
from src.char_rnn import CharRnn, CharRnnTrainer

char_rnn.device_to_use = 'cpu'

print("Loading the Encyclopedia Britannica dataset...")

britannica = open('datasets/britannica.txt').read()
britannica_alphabet = ''.join(set(britannica))

print("Loaded successfully. The alphabet for this dataset is:")
print(britannica_alphabet)
print()

print("Creating a neural net and trainer...")

britannica_model = CharRnn(britannica_alphabet, hidden_size=128, num_layers=3)
britannica_trainer = CharRnnTrainer(britannica_model, britannica, batch_size=50)

print("Doing one training step to see if it works...")

britannica_trainer.step()

print("It works. Doing another training step to see how long it takes...")

time = timeit.timeit(britannica_trainer.step, number=1)

print(f"That training step took {time:.4f} seconds.")
print()

print("Starting the training!")
print()

sampler = britannica_model.sample_randomly('. ', temperature=0.8)

for i in itertools.count(start=1):
    loss = britannica_trainer.step()
    prefix = f'{i} {loss:.4f}'
    for char in itertools.islice(sampler, 10):
        sample = char.replace('\n', f'\n{prefix} ')
        print(sample, end='', flush=True)
    #sample = ''.join(itertools.islice(sampler, 10)).replace('\n',f'\n{prefix} ')
    #print(sample, end='', flush=True)
