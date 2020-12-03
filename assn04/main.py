import nltk
import zeyrek
from nltk.corpus import stopwords
from nltk.util import ngrams
from trnlp import TrnlpWord

sw = stopwords.words('turkish')

def ReadCorpusFromFile():
    """ This function reads the first 2000 lines of the corpus from file 'corpus.csv' """
    ReadText = ""
    with open("corpus.txt", "r", encoding="utf-8") as fd:
        for i, line in enumerate(fd):
            if(i < 2000):
                ReadText = ReadText + " " + line
    return ReadText

def PrepareCorpus(corpus):
    """ This function prepares the corpus for further processing """
    # 1. Tokenize
    tokenized = nltk.word_tokenize(corpus)
    # 2. Remove Punctuation
    noPunc = [tok for tok in tokenized if tok.isalnum()]
    # 3. Remove Stopwords
    stp = list(sw)
    stp.append('')
    noStop = [w for w in noPunc if w not in stp]
    # 4. Make Lower Case
    lower = []
    for el in noStop:
        lower.append(el.lower())
    return lower

def GetFreqWithUniGram(source):
    """ This function makes unigram frequency analysis on the corpus and returns the most frequent 10 tokens """
    unigram = ngrams(source, 1)
    frequency = nltk.FreqDist(unigram)
    var = []
    i = 0
    for asd in frequency.most_common():
        if(i<10):
            var.append(asd)
            i = i+1
        else:
            break
    return var

def Main():
    ReadCorpus = ReadCorpusFromFile()
    PreparedText = PrepareCorpus(ReadCorpus)
    Variable = GetFreqWithUniGram(PreparedText)


    vars = []
    for var in Variable:
        data = str(var[0])
        length = len(data)
        vars.append(data[2:(length-3)])


    print("TAGGER |    WORD    |   LEMMA    |  POS\n-------+------------+------------+------------")
    analyzer = zeyrek.MorphAnalyzer()
    for word in vars:
        tok = TrnlpWord()
        tok.setword(word)
        print(" TRNLP |  %8s  |  %8s  | " % (word, tok.get_base), tok.get_base_type )
        tmp = analyzer.analyze(word)[0][0]
        print("ZEYREK |  %8s  |  %8s  | " % (tmp[0], tmp[1]), tmp[2])

    return True



# MAIN CODE FIELD
Main()
