import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        model_evaluations = []
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000)
                model.fit(self.X, self.lengths)

                try:
                    log_loss = model.score(self.X, self.lengths)
                except ValueError:
                    log_loss = -99999

                num_params = 2 * n_components * len(self.sequences[0][0])
                BIC = -2 * log_loss + num_params * np.log(len(self.X))
                model_evaluations.append((n_components, BIC, model))
            except ValueError:
                pass

        model_evaluations = sorted(model_evaluations, key=lambda t: -t[1])
        return model_evaluations[0][2]


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        model_evaluations = []
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000)

            try:
                model.fit(self.X, self.lengths)
            except ValueError:
                continue

            try:
                log_loss_Xi = model.score(self.X, self.lengths)
            except ValueError:
                log_loss_Xi = -99999

            log_loss_sum_Xj = 0.0
            m = 0
            for word in self.words.keys():
                if word == self.this_word:
                    continue
                word_x, word_len = self.hwords[word]

                try:
                    log_loss_Xj = model.score(word_x, word_len)
                except ValueError:
                    continue

                log_loss_sum_Xj += log_loss_Xj
                m += 1

            if m == 0:
                DIC = log_loss_Xi
            else:
                DIC = log_loss_Xi - log_loss_sum_Xj / m

            model_evaluations.append((n_components, DIC, model))

        model_evaluations = sorted(model_evaluations, key=lambda t: -t[1])
        return model_evaluations[0][2]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n_splits = min(len(self.sequences), 3)
        model_evaluations = []

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            losses = []
            model = None

            if n_splits == 1:
                model = self.base_model(n_components)
                if model is None:
                    continue
                try:
                    log_loss = model.score(self.X, self.lengths)
                except ValueError:
                    log_loss = -99999

                model_evaluations.append((n_components, log_loss, model))
                continue


            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            for train_index, test_index in kf.split(self.sequences):
                x_train, len_train = combine_sequences(train_index, self.sequences)
                x_test,  len_test = combine_sequences(test_index, self.sequences)
                try:
                    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000)
                    model.fit(x_train, len_train)
                    log_loss = model.score(x_test, len_test)
                    losses.append(log_loss)
                except ValueError:
                    pass

            if len(losses):
                model_evaluations.append((n_components, np.mean(losses), model))

        model_evaluations = sorted(model_evaluations, key=lambda t: -t[1])
        return model_evaluations[0][2]

