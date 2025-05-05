import os
import io
import string
import unicodedata
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob 
import random
import matplotlib.pyplot as plt

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
        return [self.unicode_to_ascii(line) for line in lines]

    def random_choice(self, a):
        return a[random.randint(0, len(a) - 1)]

class Dataloader:
    def __init__(self):
        self.ALL_LETTERS = string.ascii_letters + " .,;'"
        self.N_LETTERS = len(self.ALL_LETTERS)
        self.category_lines = {}
        self.all_categories = []
        self.helper_fn = helper()

    def load_data(self):
        for filename in self.helper_fn.find_files("namedata/*.txt"):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.helper_fn.read_lines(filename)
            self.category_lines[category] = lines
        return self.category_lines, self.all_categories
    
    def letter_to_tensor(self, letter):
        tensor = torch.zeros(1, self.N_LETTERS)
        tensor[0][self.ALL_LETTERS.find(letter)] = 1
        return tensor
    
    def random_training_example(self):
        category = self.helper_fn.random_choice(self.all_categories)
        line = self.helper_fn.random_choice(self.category_lines[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        return category, line, category_tensor
    
    def category_from_output(self, output):
        category_idx = torch.argmax(output).item()
        return self.all_categories[category_idx]

# Recurive Neural Network
class RecNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecNN, self).__init__()
        self.hidden_size = hidden_size
        self.V = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def encode(self, node):
        if node[0] is not None:
            return F.relu(self.V(node[0]))
        left_hidden = self.encode(node[1])
        right_hidden = self.encode(node[2])
        combined = F.relu(self.U(left_hidden) + self.W(right_hidden))
        return combined
    
    def forward(self, tree):
        hidden = self.encode(tree)
        output = self.h2o(hidden)
        return self.softmax(output)
    
class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion

    def string_to_tree(self, line):
        def build(line):
            if len(line)==1:
                return(self.dataloader.letter_to_tensor(line), None, None)
            mid = len(line)//2
            return(None, build(line[:mid]), build(line[mid:]))
        return build(line)
    
    def train(self, tree, category_tensor):
        self.model.zero_grad()
        output = self.model(tree)
        loss = self.criterion(output, category_tensor)
        loss.backward()
        self.optimizer.step()
        return output, loss.item()
    
    def predict(self, input_line):
        print(f"\n> {input_line}")
        with torch.no_grad():
            tree = self.string_to_tree(input_line)
            output = self.model(tree)
            guess = self.dataloader.category_from_output(output)
            print(guess)

    def train_model(self, n_iters=10000, plot_steps=500, print_steps=1000, plot=True):
        current_loss = 0
        all_losses = []
        for i in range(n_iters):
            category, line, category_tensor = self.dataloader.random_training_example()
            tree = self.string_to_tree(line)
            output, loss = self.train(tree, category_tensor)
            current_loss += loss

            if (i+1) % plot_steps == 0:
                all_losses.append(current_loss / plot_steps)
                current_loss = 0

            if (i+1) % print_steps == 0:
                guess = self.dataloader.category_from_output(output)
                correct = "CORRECT" if guess == category else f"WRONG ({category})"
                print(f"{i+1} : {(i+1)/n_iters*100:.2f}% {loss:.4f} {line} / {guess} {correct}")

        if plot:
            plt.figure()
            plt.plot(all_losses)
            plt.xlabel('iterations (in 1000s)')
            plt.ylabel('Loss')
            plt.show()

# Main
if __name__ == '__main__':
    dataloader = Dataloader()
    dataloader.load_data()

    hidden_size = 128
    model = RecNN(dataloader.N_LETTERS, hidden_size, len(dataloader.all_categories))
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    trainer = Trainer(model, dataloader, optimizer, criterion)
    trainer.train_model()

    while True:
        sentence = input("Input: ")
        if sentence.lower() == 'quit':
            break
        trainer.predict(sentence)





