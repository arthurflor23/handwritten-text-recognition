"""
This code is intended to analyze the prediction file (HTR project output file format)
and to compare the tokens between ground truth and predict.

A simple code is used to see how much and which tokens (character level) are most wrong by the optical model.
You can also generate a list of similarity keys.

This file is an isolated function, so use:
    `python similar_error_analysis.py` with the TXTs files to analyse in `./predicts`
"""

import os
import re
import unicodedata
from preproc import text_standardize


data_path = "./predicts"
files = os.listdir(data_path)
lines = []

for f in files:
    sentences = open(os.path.join(data_path, f), "r").read().splitlines()
    sentences = [line for line in sentences if line]

    for i in range(0, len(sentences), 2):
        text_1 = sentences[i].split()[1::]
        text_2 = sentences[i + 1].split()[1::]

        if len(text_1) > 5:
            text_1 = text_standardize(" ".join(text_1))
            text_1 = unicodedata.normalize("NFKD", text_1).encode("ASCII", "ignore").decode("ASCII")
            lines.append(" ".join(text_1.split()))

            text_2 = text_standardize(" ".join(text_2))
            text_2 = unicodedata.normalize("NFKD", text_2).encode("ASCII", "ignore").decode("ASCII")
            lines.append(" ".join(text_2.split()))

gt, dt = [], []

for i in range(0, len(lines), 2):
    if len(lines[i].split()) == len(lines[i + 1].split()):
        gt.append(re.compile(r'(.)\1{1,}', re.IGNORECASE).sub(r'\1', lines[i]))
        dt.append(re.compile(r'(.)\1{1,}', re.IGNORECASE).sub(r'\1', lines[i + 1]))

words = []

for g, d in zip(gt, dt):
    tokens_1 = g.split()
    tokens_2 = d.split()

    for y, x in zip(tokens_1, tokens_2):
        if y != x and len(y) == len(x):
            words.append([y, x])

char_dict = dict()

for w in words:
    list_1 = list(w[0])
    list_2 = list(w[1])

    for i in range(len(list_1)):
        if list_1[i] != list_2[i]:
            try:
                char_dict[list_1[i]].append(list_2[i])
            except KeyError:
                char_dict[list_1[i]] = []
                char_dict[list_1[i]].append(list_2[i])

similarity_list = []

for x in char_dict:
    occur = len(char_dict[x])
    tokens = list(set(char_dict[x]))
    similarity_list.append([x, occur, tokens])

similarity_list = sorted(similarity_list, key=lambda x:x[1], reverse=True)
token_list = []

for index, w in enumerate(similarity_list):
    print(f"{w[0]} = {w[1]} occurrences.\nTokens: {w[2]}\n")
    y = w[0]

    for x in w[2]:
        if y.lower() == x.lower() or [y, x] in token_list:
            continue

        token_list.append([y, x])

print("Similarity list:", sorted(token_list), "\n")
print("Items:", len(token_list))
