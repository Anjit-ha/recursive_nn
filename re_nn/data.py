# Dataloader
import os
import string
import torch
from utils import helper

class Dataloader:
    def __init__(self):
        self.ALL_LETTERS = string.ascii_letters + " .,;'"
        self.N_LETTERS = len(self.ALL_LETTERS)
        self.category_lines = {}
        self.all_categories = []
        self.helper_fn = helper()

    def load_data(self):
        for filename in self.helper_fn.find_files("data/*.txt"):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.helper_fn.read_lines(filename)
            self.category_lines[category] = lines
        return self.category_lines, self.all_categories

    def letter_to_index(self, letter):
        return self.ALL_LETTERS.find(letter)

    def letter_to_tensor(self, letter):
        tensor = torch.zeros(1, self.N_LETTERS)
        index = self.letter_to_index(letter)
        if index != -1:
            tensor[0][index] = 1
        return tensor

    def line_to_tensor(self, line):
        tensor = torch.zeros(len(line), 1, self.N_LETTERS)
        for li, letter in enumerate(line):
            index = self.letter_to_index(letter)
            if index != -1:
                tensor[li][0][index] = 1
        return tensor

    def random_training_example(self):
        category = self.helper_fn.random_choice(self.all_categories)
        line = self.helper_fn.random_choice(self.category_lines[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        line_tensor = self.line_to_tensor(line)
        return category, line, category_tensor, line_tensor

    def category_from_output(self, output):
        category_idx = torch.argmax(output).item()
        return self.all_categories[category_idx]
