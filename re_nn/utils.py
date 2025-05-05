import os
import io
import string
import unicodedata
import glob
import random

class helper:
    def __init__(self):
        self.ALL_LETTERS = string.ascii_letters + " .,;'"

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.ALL_LETTERS
        )

    def find_files(self, path):
        return glob.glob(path)

    def read_lines(self, filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicode_to_ascii(l) for l in lines]

    def random_choice(self, l):
        return random.choice(l)
