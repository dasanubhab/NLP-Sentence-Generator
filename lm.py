# imports go here
import sys
import numpy as np
"""
Don't forget to put your name and a file comment here
Anubhab Das
"""


# Feel free to implement helper functions

class LanguageModel:
    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing):
        """Initializes an untrained LanguageModel
        Parameters:
          n_gram (int): the n-gram order of the language model to create
          is_laplace_smoothing (bool): whether or not to use Laplace smoothing
        """
        # length of each ngram we look at
        self.n_gram = n_gram
        # defines what way we should smooth the probability distribution
        self.is_laplace_smoothing = is_laplace_smoothing
        # dictionary to keep track of the distribution of ngrams
        self.counts = {}
        # list of every kind of token
        self.unique_toks = []
        pass

    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Parameters:
          training_file_path (str): the location of the training data to read

        Returns:
        None
        """
        # Unpack contents of file so that data can be processed
        f = open(training_file_path, "r", encoding="utf-8")
        content = f.read()  # read all contents
        f.close()  # close the file when you're done
        toks = content.split()
        counts = {}
        # run through the data keep tracking of how much each token there is in
        # a dictionary
        for index in range(len(toks)):
            if counts.get(toks[index], 0) == 0:
                counts[toks[index]] = 1
            else:
                counts[toks[index]] += 1
        # variable to keep track of if there is a unique token
        a_unique = False
        for index in range(len(toks)):
            # loop through replacing tokens where only one of its kind exists
            if counts[toks[index]] == 1:
                toks[index] = self.UNK
                if not a_unique:
                    self.unique_toks.append(self.UNK)
                    a_unique = True
            else:
                if toks[index] not in self.unique_toks:
                    self.unique_toks.append(toks[index])
        # loop through creating n-grams for every subarray of length n
        # then make it a tuple and add it to the dict of grams (self.count)
        for i in range(len(toks)-self.n_gram+1):
            current_n_gram = []
            for f in range(self.n_gram):
                current_n_gram.append(toks[i+f])
            if tuple(current_n_gram) not in self.counts:
                self.counts[tuple(current_n_gram)] = 1
            else:
                self.counts[tuple(current_n_gram)] += 1
    # test function for testing data sets from given filepath

    def test(self, testing_file_path):
        f = open(testing_file_path, "r", encoding="utf-8")
        content = f.read()  # read all contents
        f.close()  # close the file when you're done
        toks = content.split()
        # display filepath
        print("test corpus: ", testing_file_path)
        left = 0
        right = 0
        # keep sentences in a list for data operations later
        test_sentences = []
        sent = ""
        # find sentences by seeing where "<s>" and "</s>" meet
        for index in range(len(toks) - 1):
            sent += toks[index] + " "
            if toks[index] == self.SENT_END and toks[index+1] == self.SENT_BEGIN:
                left = right
                right = index+1
                test_sentences.append(sent[:-1])
                sent = ""
        test_sentences.append(sent+toks[-1])
        # simply use length of list for num of sentences
        print("Num of test sentences: ", len(test_sentences))
        avg = 0
        probabilities = []
        # score each sentence to find respective probabilities
        for sentence in test_sentences:
            probabilities.append(self.score(sentence))
        # use numpy to do standard operations average and stdev
        print("Average probability: ", np.average(probabilities))
        print("Standard Deviation: ", np.std(probabilities))

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
        Parameters:
          sentence (str): a sentence with tokens separated by whitespace to calculate the score of

        Returns:
          float: the probability value of the given string for this model
        """
        # split sentence by whitespace
        sentence_sequence = sentence.split()
        # find uniqe tokens to the sentence
        for index in range(len(sentence_sequence)):
            if sentence_sequence[index] not in self.unique_toks:
                sentence_sequence[index] = self.UNK
        # probability which will be found through a chain of probabilites, so
        # starts at 1
        p = 1
        if self.is_laplace_smoothing:
            for i in range(len(sentence_sequence)-self.n_gram+1):
                # use subarry to find n-gram
                current_n_gram = sentence_sequence[i:self.n_gram+i]
                # find how many they are for the numerator
                # add 1 because laplacian
                numerator = self.counts.get(tuple(current_n_gram), 0) + 1
                # use function for denominator
                denominator = self.find_denominator(current_n_gram)
                # add length since its laplacian
                denominator += len(self.unique_toks)
                p *= numerator/denominator
        else:
            for i in range(len(sentence_sequence)-self.n_gram+1):
                current_n_gram = sentence_sequence[i:self.n_gram+i]
                # find how many there are for numerator
                numerator = self.counts.get(tuple(current_n_gram), 0)
                # use function to find denominator
                denominator = self.find_denominator(current_n_gram)
                p *= numerator/denominator
        return p

    def find_denominator(self, current_n_gram):
        denominator = 0
        # denominator is found through iterating through all possible tokens
        # and combining them with the last n-1 tokens to create a probability
        # distribution
        for tok in self.unique_toks:
            current_n_gram[-1] = tok
            denominator += self.counts.get(tuple(current_n_gram), 0)
        return denominator

    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          str: the generated sentence
        """
        # start sentence off with max(n-1, 1) <s> tokens
        sentence = [self.SENT_BEGIN]
        for i in range(self.n_gram-2):
            sentence.append(self.SENT_BEGIN)
        # generate tokens randomly until </s> is generated
        while sentence[-1] != self.SENT_END:
            # if n_gram > 2 create a fragment of the last n-1 tokens
            if self.n_gram != 1:
                frag = sentence[-self.n_gram+1:]
            else:
                frag = []
            total = 0
            frag.append("")
            # prob distribution array
            parray = []
            # iterate through all possible tokens and combine them with n-1
            # tokens in a tuple to create a probability distribution
            for tok in self.unique_toks:
                # when working in sentences no intermediate tokens can be <s>
                if tok != self.SENT_BEGIN:
                    frag[-1] = tok
                    parray.append(self.counts.get(tuple(frag), 0))
                    total += parray[-1]
                else:
                    parray.append(0)
            if total != 0:
                for i in range(len(parray)):
                    parray[i] /= total
            # using numpy we can generate random element using the parray
            sentence.extend(np.random.choice(self.unique_toks, 1, p=parray))
        for i in range(self.n_gram-2):
            sentence.append(self.SENT_END)
        # make sentence out of array of tokens
        finalsentence = ' '.join(sentence)
        return finalsentence

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
        Parameters:
          n (int): the number of sentences to generate

        Returns:
          list: a list containing strings, one per generated sentence
        """
        # generate sentences and append to list
        sentences = []
        for num in range(n):
            sentences.append(self.generate_sentence())
        return sentences

    def perplexity(self, test_sequence):
        """
                Measures the perplexity for the given test sequence with this trained model. 
                As described in the text, you may assume that this sequence may consist of many sentences "glued together".

        Parameters:
            test_sequence (string): a sequence of space-separated tokens to measure the perplexity of
        Returns:
            float: the perplexity of the given sequence
        """
        # perplexity is a simple computation on the score that we calculate
        n = len(test_sequence.split())
        probability = self.score(test_sequence)
        # make sure to avoid dividing by 0
        if n == 0 or probability == 0:
            return 0
        # this simple expression is quivalent to the perplexity
        return (1/probability)**(1/n)


def main():
    # filepath arguments
    training_path = sys.argv[1]
    testing_path1 = sys.argv[2]
    testing_path2 = sys.argv[3]
    # create unigram and bigram models (also works for n >= 2)
    unigram_model = LanguageModel(1, True)
    bigram_model = LanguageModel(2, True)
    # train models on training set
    unigram_model.train(training_path)
    bigram_model.train(training_path)
    # generate sentences
    unigram_sentences = unigram_model.generate(50)
    print("Model: unigram, laplace smoothed ")
    print("sentences:")
    for sent in unigram_sentences:
        print(sent)
    # test using function above
    unigram_model.test(testing_path1)
    unigram_model.test(testing_path2)
    # generate sentences
    bigram_sentences = bigram_model.generate(50)
    print("Model: bigram, laplace smoothed ")
    print("sentences:")
    for sent in bigram_sentences:
        print(sent)
    # test using functions above
    bigram_model.test(testing_path1)
    bigram_model.test(testing_path2)
    print("perplexity: ", bigram_model.perplexity(
        "<s> i don't matter i want to ten dollars maybe if you show me the "
        + "cost to go for breakfast </s> <s> i'd like to eat a lot of the "
        "information about a fancy but the <UNK> me information about five " +
        "minutes </s>"))


if __name__ == '__main__':

    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) != 4:
        print(
            "Usage:", "python hw2_lm.py training_file.txt testingfile1.txt testingfile2.txt")
        sys.exit(1)

    main()
