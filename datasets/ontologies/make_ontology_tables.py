import collections
import os
import random

import latextable
import texttable


def load_files(files):
    words_per_file = collections.defaultdict(list)
    for filename in files:
        cls_name = os.path.splitext(filename)[0].removeprefix('Tax-')
        with open(filename) as file:
            for line in map(str.strip, file):
                if not line:
                    continue
                words_per_file[cls_name].append(line)
    return words_per_file


def make_table(files):
    words_per_class = load_files(files)

    table = texttable.Texttable()
    table.set_deco(texttable.Texttable.HEADER)
    table.header(['Class Name', 'Example Words'])

    n = 10

    entries = sorted(words_per_class)
    for entry in entries:
        words = words_per_class[entry]
        if len(words) < n:
            selection = words
        else:
            selection = random.sample(words, k=n)
        table.add_row([entry, ', '.join(sorted(selection))])

    print(table.draw())
    print()
    print(latextable.draw_latex(table))


def main():
    ontology = []
    lexical = []
    for filename in os.listdir():
        if not filename.endswith(('.txt', '.csv')):
            continue
        if 'Tax' in filename:
            lexical.append(filename)
        else:
            ontology.append(filename)
    make_table(ontology)
    make_table(lexical)

main()
