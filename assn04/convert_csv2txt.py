import csv
import urllib.request

csv_file = 'texts.csv'
corpus_csv = 'corpus.csv'
corpus_text = 'corpus.txt'

print("Downloading the corpus...")
url = "https://github.com/selimfirat/bilkent-turkish-writings-dataset/raw/master/data/texts.csv"
urllib.request.urlretrieve(url, csv_file)

print("Converting the corpus to text...")
fi = open(csv_file, 'rb')
data = fi.read()
fi.close()
fo = open(corpus_csv, 'wb')
fo.write(data.replace(b'\x00', b''))
fo.close()

with open(corpus_text, "w") as ofd:
    with open(corpus_csv, "r") as ifd:
        readData = csv.reader(ifd)
        i = 0
        for row in readData:
            if(i != 0):
                ofd.write(row[1])
            i = i + 1

import os
os.remove("texts.csv")
os.remove("corpus.csv")

print("Done!")
