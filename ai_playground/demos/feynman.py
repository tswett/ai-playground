from ai_playground.char_rnn import CharRnn, CharRnnTrainer

if __name__ == '__main__':
    print('First, we generate a char-rnn and ask it to predict each following ' +
          'character in a famous quote from Richard Feynman.')
    print()

    feynman_model = CharRnn(' #-.FRTacdefghilmnoprstuy')
    prediction = feynman_model.predict_each_next('#The important thing is to not fool yourself. -- Richard Feynman')
    print()

    print('The text that it predicted was:')
    print()

    print(prediction)
    print()

    print("That wasn't very good. Let's train it to do better.")
    print()

    feynman_trainer = CharRnnTrainer(feynman_model, '#The important thing is to not fool yourself. -- Richard Feynman#')

    steps = 100
    for n in range(steps):
      loss = feynman_trainer.step()
      if n % 10 == 0 or n == steps - 1:
        prediction = feynman_model.predict_each_next('#The important thing is to not fool yourself')
        print(f'{n:02d} {float(loss):.4f} {prediction}')

    print()

    print('That looks a lot better! Now the neural net can probably print the ' +
          "whole quote all on its own. Let's try:")
    print()

    print(''.join(feynman_model.sample('#', 63)))
    print()

    print("Now, just for fun, let's try sampling randomly instead of " +
          "deterministically. We'll probably end up with jumbled nonsense.")
    print()

    for char in feynman_model.sample_randomly('#', 500):
      print(char.replace('#','\n'), end='', flush=True)
    print()
