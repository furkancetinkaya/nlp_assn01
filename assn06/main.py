import json
import string
import os
from nltk.corpus import stopwords
from trnlp.morphology import TrnlpWord
from nltk import word_tokenize
stop_words = set(stopwords.words('turkish'))


class Dictionary:
    """ Reads the dictionary.json and holds its entries. """

    def __init__(self) -> None:
        with open('dictionary.json', 'r', encoding='utf-8') as dictf:
            self._dictionary = json.load(dictf)

    def get_entry(self, word, pos):
        """ Get a dictionary entry for a specific POS tagged word """
        # Search for the entry
        for entry in self._dictionary:
            if entry['_word'] == word and entry['_pos'] == pos:
                return entry
        return None


def compute_overlap(signature, context):
    """ Computing the number of overlaps within dictionary gloss/example words and words given by the corpus. """
    # Remove stopwords, punctuation marks, and lower the cases
    filtered_signature = [w for w in signature if w not in stop_words and w not in string.punctuation]
    filtered_context = [w for w in context if w not in stop_words and w not in string.punctuation]
    lowered_signature = [low.lower() for low in filtered_signature]
    lowered_context = [low.lower() for low in filtered_context]

    # find the overlapped words within signature and context
    overlapped = []
    for s in lowered_signature:
        for c in lowered_context:
            if s == c:
                overlapped.append(c)

    # Remove multiple occurrences of the words
    uniquelist = []
    for el in overlapped:
        if el not in uniquelist:
            uniquelist.append(el)

    return len(uniquelist)


def get_sense_gloss_words(sense):
    """ Read an write a specific sense's words to a list with lemmatizing them """
    # Split gloss and example to words
    glosses = word_tokenize(sense['_gloss'])
    glosses += word_tokenize(sense['_example'])

    # Lemmatize the splitted words
    lemmatized = []
    for w in glosses:
        obj = TrnlpWord()
        obj.setword(w)
        lemmatized.append(obj.get_base)
    return lemmatized


class LESK:
    """ Simplified Lesk Algorithm implemented """

    def __init__(self) -> None:
        self._dictionary = Dictionary()

    def lesk(self, word, sentence, pos):
        # Tokenize the sentence
        tokenized = word_tokenize(sentence)

        # Remove stopwords, punctuation marks, and lemmatize sentence words
        cleared = [tok for tok in tokenized if tok not in stop_words and tok not in string.punctuation]
        lowered = [low.lower() for low in cleared]
        context = []
        for ww in lowered:
            obj = TrnlpWord()
            obj.setword(ww)
            context.append(obj.get_base)
        lowed = [low.lower() for low in context]
        context = lowed

        # Starting point of the LESK Algorithm
        best_sense = self.get_best_sense(word, pos)     # Get the best meaning of the ambiguous word
        max_overlap = 0                                 # İnitialize max_overlap
        senses = self.get_all_senses(word, pos)         # Get all alternative meanings of the word

        for sense in senses:
            signature = get_sense_gloss_words(sense)        # Get the dictionary definition/example words
            overlap = compute_overlap(signature, context)   # Compute the number of overlappings
            if overlap > max_overlap:         # If the number of overlap is greater than the previous one,
                max_overlap = overlap         # --> update max_overlap
                best_sense = sense            # --> update best meaning
        return best_sense

    def get_all_senses(self, word, pos):
        """ Get all senses for a specified word in the dictionary """
        senses = []
        for sense in self._dictionary.get_entry(word, pos)["_senses"]:
            senses.append(sense)
        return senses

    def get_best_sense(self, word, pos):
        """ Read the best sense of a word from the dictionary """
        entry = self._dictionary.get_entry(word, pos)
        return entry['_senses'][entry['_best_sense']]


def main() -> int:
    os.system("clear")

    corpus = [
        'Düşük siyah bıyıklarına, sakalına pek az kır düşmüş olan Selim Paşa, karısından çok genç görünüyordu.',
        'Düğüne kimlerin çağrıldığı anlaşılmaz, ne hediye gönderileceği de belli olmaz. Olmaz ama hepsi çağrılmıştır, \
hepsi de kırıp sarar, birer hediye alır yollar.',
        'İnce ve yüksek bir sanat eseri olan saz da milliyetimizin bir hususiyetidir.',
        'Sazların arasındaki ördek yuvaları, tilkilerin istilasına uğramıştı.'
    ]

    mylesk = LESK()

    test1 = mylesk.lesk('kır', corpus[0], 'n')
    test2 = mylesk.lesk('kır', corpus[1], 'v')
    test3 = mylesk.lesk('saz', corpus[2], 'n')
    test4 = mylesk.lesk('saz', corpus[3], 'n')

    print()

    print("Corpus: ", corpus[0])
    print("--> Ambiguous Word : kır")
    print("--> Meaning        : ", test1['_gloss'], end="\n\n")

    print("Corpus: ", corpus[1])
    print("--> Ambiguous Word : kır")
    print("--> Meaning        : ", test2['_gloss'], end="\n\n")

    print("Corpus: ", corpus[2])
    print("--> Ambiguous Word : saz")
    print("--> Meaning        : ", test3['_gloss'], end="\n\n")

    print("Corpus: ", corpus[3])
    print("--> Ambiguous Word : saz")
    print("--> Meaning        : ", test4['_gloss'])

    return 0


main()
