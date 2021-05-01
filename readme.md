This is my "AI playground," a collection of scripts and whatnot for playing with
machine learning models.

## Char-RNN demos and commands

### Feynman quote demo

The Feynman quote demo trains a small neural net on a quote from Richard
Feynman. Simply run:

    python -m ai_playground.demos.feynman

The demo will start by creating a neural net and asking it to produce some text.
The initial prediction will be total garbage, since the neural net hasn't been
trained yet. Next, it will spend a short amount of time training the neural net,
and outputting its progress.

During the training, the demo does a "predict-each-next" sample from the neural
net, meaning that it asks the neural net for predictions, one character at a
time, but after each step, it provides the neural net with the correct answer,
regardless of what its prediction was.

After the training, the demo does a deterministic sample, where it repeatedly
asks the neural net for character predictions, chooses the character the neural
net thought was most likely, prints that character, and feeds that same
character back to the neural net. If the neural net guesses the wrong letter,
then that incorrect guess will be fed back into the neural net, and so
subsequent guesses are likely to be wrong, too.

Finally, the demo does a random sample. This works just like the above, except
that the demo no longer chooses the character the neural net thinks is most
likely. Instead, it chooses a character randomly, according to what the neural
net thinks is more or less likely.

### Encyclopedia Britannica demo

The Britannica demo creates a neural net and trains it on a section of the
Encyclopedia Britannica. Run:

    python -m ai_playground.demos.britannica

The demo will both train and sample from the neural net at the same time, so
that you get to constantly see what the neural net has learned.

Initially, the output will seem like completely random characters. Very quickly,
the neural net will start to output mostly lowercase letters and spaces, but it
will still be a completely jumbled mess. After about 500 training steps, the
neural net will be outputting a lot of short, common words (like "to the for in
the of as"), and the rest of the text will be phonologically plausible nonsense
(like "fever linendenpiom conturam Argic was dufred"). You can see the number of
training steps that the neural net has completed in the left-hand column.

If you let the demo run for a while longer (maybe around 5,000 steps), it will
start to output mostly real English words, but the order won't make any sense.
After a while, it will also start to wrap lines more or less consistently.

If you let it run for a very long time, then it will start to produce sentences
that are mostly grammatically correct, but it will probably never stop making
grammatical mistakes. Some of the text that it outputs may even seem to make
some kind of sense, but it's very, very, very, very unlikely that this neural
net will ever learn to write anything coherent and meaningful.

With this command, there is no way to save the neural net's progress!

### Sampling from a pre-trained neural net

The playground comes with an Encyclopedia Britannica model that you can sample
from right away. Just run:

    PYTHONPATH=ai_playground python -m ai_playground.train_char_rnn --model_name models/britannica-2021-04-17-19-59-12 --sample_chars 2000 --save_every 1000000

The neural net will start to produce some text that will look similar to this:

    1 1.1970 Pharis were far applied that had not already and a fine love, any the may
    1 1.1970 name in lighted. The earlier ruler of the emperor was known as the
    1 1.1970 confusion of the _Potaissamidatis_ and far less poetry pieces of its
    1 1.1970 expression with the lifetiment of the writers in the ancient instrudent
    1 1.1970 of such as the later eggs of Épinal.

(The PYTHONPATH bit at the beginning is needed because this is an older model
file which uses an import path which is no longer correct.)

### Creating and training your own neural net

To train a neural net of your own, start by finding a text file you like and
putting it in the `datasets` directory. Name it something like `mydata.txt`.
Then run:

    python -m ai_playground.create_char_rnn --dataset datasets/mydata.txt --hidden_size 128 --num_layers 3 --model_name models/mymodel

This will create a model file called `models/mymodel`, but won't do any
training.

Next, run:

    python -m ai_playground.train_char_rnn --model_name models/mymodel --save_model_name models/mymodel --save_every 500

(It's safe to use `models/mymodel` as the output filename because
`train_char_rnn.py` will append the date and time to this filename, in order to
avoid overwriting the existing file.)

Note that when you create the neural net, the path to the dataset will be saved.
The `train_char_rnn.py` program will then read from that path in order to train
the neural net.

You can change the training data by passing another `--dataset` parameter to
`train_char_rnn.py`. Currently, `train_char_rnn.py` is unable to train on new
data that contains a character which is not present in the old data.

Of course, the model that you pass to `train_char_rnn.py` can be either an
untrained model created by `create_char_rnn.py`, or a trained model created by
`train_char_rnn.py`.
