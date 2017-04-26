import warnings
import numpy as np
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    test_all_data = test_set.get_all_Xlengths()
    for word_index in test_all_data.keys():
        x, xlen = test_all_data[word_index]
        prob_result = {}
        for guess_word in models.keys():
            guess_model = models[guess_word]
            try:
                loss = guess_model.score(x, xlen)
                prob_result[guess_word] = loss
            except:
                prob_result[guess_word] = -99999

        best_guess = list(prob_result.keys())[np.argmax(list(prob_result.values()))]

        probabilities.append(prob_result)
        guesses.append(best_guess)

    return probabilities, guesses

