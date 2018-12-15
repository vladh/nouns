import ast
import matplotlib.pyplot as plt


FILE = 'results/20181215-length-optimisation-2.txt'


def get_data():
    with open(FILE, 'r') as f:
        lines = [line.strip() for line in f]

    line = lines[0]
    data = ast.literal_eval(line)
    return data


def plot_data(data):
    fig, ax = plt.subplots()
    x = [d[0][2] for d in data]
    y = [d[1] for d in data]
    ax.plot(x, y)
    ax.set_xlabel('Number of items')
    ax.set_ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    data = get_data()
    plot_data(data)
