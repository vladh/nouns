import ast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


FILE = 'results/20181215T171400-optimisation-1.txt'


def get_data():
    with open(FILE, 'r') as f:
        lines = [line.strip() for line in f]

    line = lines[0]
    data = ast.literal_eval(line)
    return data


def plot_data(data):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = [d[0][0] for d in data]
    y = [d[0][1] for d in data]
    z = [d[1] for d in data]
    ax.plot_trisurf(x, y, z)
    # ax.set_title('Parameter optimisation')
    ax.set_xlabel('Frequency set minimum magnitude')
    ax.set_ylabel('Frequency set minimum proportion')
    ax.set_zlabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    data = get_data()
    plot_data(data)
