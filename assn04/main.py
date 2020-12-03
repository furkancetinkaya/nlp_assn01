import nltk
import zeyrek
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from trnlp import TrnlpWord, word_to_number


sw = stopwords.words('turkish')

def readcorpus():
    ReadText = ""
    with open("corpus.csv", "r", encoding="utf-8") as fd:
        for i, line in enumerate(fd):
            if(i < 1000):
                ReadText = ReadText + " " + line
    return ReadText

def preparetext(corpus):
    # 1. Tokenize
    tokenized = nltk.word_tokenize(corpus)
    # 2. Remove Punctuation
    noPunc = [tok for tok in tokenized if tok.isalnum()]
    # 4. Remove Stopwords
    stp = list(sw)
    stp.append('')
    noStop = [w for w in noPunc if w not in stp]
    lower = []
    for el in noStop:
        lower.append(el.lower())
    return lower

def GetFreqWithUniGram(source):
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
    ReadCorpus = readcorpus()
    PreparedText = preparetext(ReadCorpus)
    Variable = GetFreqWithUniGram(PreparedText)

    vars = []
    for var in Variable:
        dd = str(var[0])
        length = len(dd)
        vars.append(dd[2:(length-3)])

    print("\n----------------------- TRNLP TAGGER -----------------------\n")
    for asd in vars:
        obj = TrnlpWord()
        obj.setword(asd)
        #print("%10s  --> " % (asd), obj.get_base_type)
        print("Word: %8s  --> " % (asd), "Lemma: %8s  --> " % (obj.get_base), "POS: ", obj.get_base_type)
    
    print("\n----------------------- ZEYREK TAGGER -----------------------\n")
    analyzer = zeyrek.MorphAnalyzer()
    for asd in vars:
        print("Word: %8s  --> " % (analyzer.analyze(asd)[0][0][0]), "Lemma: %8s  --> " % (analyzer.analyze(asd)[0][0][1]), "POS: ", analyzer.analyze(asd)[0][0][2])
    return True


Main()
