import csv
import string

def parse(s):
    if s.isalpha():
        return s
    if "." in s:
        return float(s)
    return int(s)


def to_num(letter):
    return string.ascii_uppercase.index(letter)


with open('../data sets/LetterDataSetOrig.csv') as fd:
    data = [tuple([parse(x) for x in line]) for line in csv.reader(fd)]

mod_data = [line[1:]+(to_num(line[0]),) for line in data]

with open("../data sets/LetterDataSet.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(mod_data)
