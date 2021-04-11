from abc import abstractmethod
import torch
from torch import nn, Tensor

device_to_use = None

def send(thing):
  global device_to_use
  if device_to_use is None:
    if torch.cuda.is_available():
      print('send: Using the GPU')
      device_to_use = 'gpu'
    else:
      print('send: Using the CPU')
      device_to_use = 'cpu'

  if device_to_use == 'gpu':
    return thing.cuda()
  else:
    return thing

class TokenSequencePredictor:
  @abstractmethod
  def forward(self, input: Tensor, memory: Tensor = None) -> (Tensor, Tensor):
    """Given an input tensor and an optional memory tensor, make a prediction.

    Keyword arguments:

    input -- Tensor of shape (batch_size, length) where each batch contains a
             sequence of input tokens. Alternatively, tensor of shape
             (batch_size,) where each batch contains a single input token.
    memory -- Memory (state) tensor, whose shape could vary.

    Returns a tuple (output, new_memory).

    output -- Tensor of shape (batch_size, alphabet_size) where each batch
              contains a probability distribution over the possible tokens.
    new_memory -- Memory (state) tensor, whose shape could vary.
    """

    raise NotImplementedError

class CharRnnCore(nn.Module):
  def __init__(self, alphabet_size: int, hidden_size: int, num_layers: int):
    super().__init__()

    self.alphabet_size = alphabet_size
    self.hidden_size = hidden_size

    self.encoder = nn.Embedding(alphabet_size, hidden_size)
    self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
    self.decoder = nn.Linear(hidden_size, alphabet_size)

    send(self)

  def forward(self,
              input: ('batch_size', 'length'),
              memory: ('batch_size', 'num_layers', 'hidden_size') = None
              ) -> (('batch_size', 'length', 'alphabet_size'),
                    ('batch_size', 'num_layers', 'hidden_size')):

    input_encoded: ('batch_size', 'length', 'hidden_size') = self.encoder(input)

    output_encoded: ('batch_size', 'length', 'hidden_size')
    new_memory: ('batch_size', 'num_layers', 'hidden_size')
    output_encoded, new_memory = self.rnn(input_encoded, memory)

    output: ('batch_size', 'length', 'alphabet_size')
    output = nn.Softmax(2)(self.decoder(output_encoded))

    return output, new_memory

class CharRnn(nn.Module, TokenSequencePredictor):
  def __init__(self, alphabet: str, hidden_size: int = 100, num_layers: int = 1):
    super().__init__()

    self.alphabet = alphabet
    alphabet_size = len(alphabet)

    self.core = CharRnnCore(alphabet_size, hidden_size, num_layers)

    send(self)

  def forward(self, input, memory = None) -> (('batch_size', 'alphabet_size'), ('batch_size', 'hidden_size')):
    if type(input) == Tensor:
      if input.dim() == 1:
        return self.forward_single_char_tensor(input, memory)
      elif input.dim() == 2:
        return self.forward_string_tensor(input, memory)
      else:
        raise TypeError(f'expected a Tensor of dim 1 or 2, received a tensor of dim {input.dim()}')
    elif type(input) == str:
      return self.forward_string(input, memory)
    else:
      raise TypeError(f'expected a Tensor or a str, received a value of type {type(input)}')

  def forward_single_char_tensor(self,
                                 input: ('batch_size'),
                                 memory: ('batch_size', 'num_layers', 'hidden_size') = None
                                 ) -> (('batch_size', 'alphabet_size'), ('batch_size', 'num_layers', 'hidden_size')):

    output, new_memory = self.core(input[:, None], memory)
    return output[:, 0, :], new_memory

  def forward_string_tensor(self,
                            input: ('batch_size', 'length'),
                            memory: ('batch_size', 'num_layers', 'hidden_size') = None
  ) -> (('batch_size', 'length', 'alphabet_size'), ('batch_size', 'num_layers', 'hidden_size')):

    return self.core(input, memory)

  def forward_string(self,
                     input: str,
                     memory: ('batch_size', 'hidden_size') = None
  ) -> (('batch_size', 'length', 'alphabet_size'), ('batch_size', 'num_layers', 'hidden_size')):
    input_tensor: ('length',) = self.str_to_indices(input)
    return self.forward_string_tensor(input_tensor[None, :], memory)

  def char_to_index(self, char: str) -> int:
    return self.alphabet.find(char)

  def char_to_one_hot(self, char: str) -> ('alphabet_size',):
    output = send(torch.zeros(len(self.alphabet)))
    output[self.char_to_index(char)] = 1

    return output

  def string_tensor_to_one_hot(self, input: ('batch_size', 'length,')) -> ('batch_size', 'length', 'alphabet_size'):
    output = send(torch.zeros(input.size(0), input.size(1), len(self.alphabet)))
    for i in range(input.size(0)):
      for j in range(input.size(1)):
        output[i, j, input[i, j]] = 1

    return output

  def str_to_indices(self, input: str) -> ('length',):
    return send(Tensor([self.char_to_index(char) for char in input]).to(torch.long))

  def str_to_one_hot(self, input: str) -> ('length', 'alphabet_size'):
    one_hot_tensors: ('alphabet_size',) = [self.char_to_one_hot(c) for c in input]
    input_tensor: ('length', 'alphabet_size') = torch.stack(one_hot_tensors)
    return input_tensor

  def predict_each_next(self, input: str) -> str:
    predictions = []
    memory: ('batch_size', 'hidden_size') = None

    for c in input:
      input_tensor: ('batch_size') = send(torch.LongTensor([self.char_to_index(c)]))

      prediction_tensor: ('batch_size', 'alphabet_size')
      prediction_tensor, memory = self.forward(input_tensor, memory)
      prediction = torch.argmax(prediction_tensor[0,:])
      predictions.append(prediction)

    return ''.join([self.alphabet[i] for i in predictions])

  def sample(self, input: str):
    prediction: ('batch_size', 'alphabet_size') = None
    memory: ('batch_size', 'hidden_size') = None

    for c in input:
      input_tensor: ('batch_size') = send(torch.LongTensor([self.char_to_index(c)]))
      prediction, memory = self.forward(input_tensor, memory)

    while True:
      c = self.alphabet[torch.argmax(prediction.squeeze(dim=0))]
      yield c

      input_tensor: ('batch_size') = send(torch.LongTensor([self.char_to_index(c)]))
      prediction, memory = self.forward(input_tensor, memory)

  def sample_randomly(self, input: str, temperature: float = 1.0):
    prediction: ('batch_size', 'alphabet_size') = None
    memory: ('batch_size', 'hidden_size') = None

    for c in input:
      input_tensor: ('batch_size') = send(torch.LongTensor([self.char_to_index(c)]))
      prediction, memory = self.forward(input_tensor, memory)

    while True:
      c = self.alphabet[torch.multinomial(prediction.squeeze(dim=0)**(1/temperature), num_samples=1)]
      yield c

      input_tensor: ('batch_size') = send(torch.LongTensor([self.char_to_index(c)]))
      prediction, memory = self.forward(input_tensor, memory)
