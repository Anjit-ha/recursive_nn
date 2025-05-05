import torch
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion

    def string_to_tree(self, line):
        def build(line):
            if len(line) == 0:
                return None
            if len(line) == 1:
                return {'value': line[0], 'left': None, 'right': None}
            mid = len(line) // 2
            return {
                'value': line[mid],
                'left': build(line[:mid]),
                'right': build(line[mid + 1:])
            }
        return build(line)

    def train(self, tree, category_tensor):
        self.model.zero_grad()
        output = self.model(tree)
        loss = self.criterion(output, category_tensor)
        loss.backward()
        self.optimizer.step()
        return output, loss.item()

    def predict(self, input_line):
        print(f'\n> {input_line}')
        with torch.no_grad():
            tree = self.string_to_tree(input_line)
            output = self.model(tree)
            guess = self.dataloader.category_from_output(output)
            print(f'= {guess}')

    def train_model(self, n_iters=100000, plot_steps=500, print_steps=1000, plot=True):
        current_loss = 0
        all_losses = []
        for i in range(n_iters):
            category, line, category_tensor, line_tensor = self.dataloader.random_training_example()
            tree = self.string_to_tree(line)
            output, loss = self.train(tree, category_tensor)
            current_loss += loss

            if (i + 1) % plot_steps == 0:
                all_losses.append(current_loss / plot_steps)
                current_loss = 0

            if (i + 1) % print_steps == 0:
                guess = self.dataloader.category_from_output(output)
                correct = '✓' if guess == category else f'✗ ({category})'
                print(f'{i + 1} {loss:.4f} {line} / guess = {guess} {correct}')

        if plot:
            plt.figure()
            plt.plot(all_losses)
            plt.title('Loss over time')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.show()
