import os
import matplotlib.pyplot as plt


def draw(table):
    plt.title('Training Loss')
    plt.xlabel('#iterations')
    plt.ylabel('total loss')
    x, y = zip(*table)
    sub_x = x[0::25]
    sub_y = y[0::25]
    new_x = [int(x) for x in sub_x]
    new_y = [float(x) for x in sub_y]
    plt.plot(new_x, new_y)

    plt.savefig('loss.jpg')


def main():
    logfile = 'train.log'
    iteration = []
    total_loss = []
    with open(logfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            split_line = line.split()
            if line.find("iter:") == -1:
                pass
            else:
                try:
                    idx = split_line.index("iter:")
                    iteration.append(split_line[idx+1])
                except Exception:
                    pass

            if line.find("total_loss:") == -1:
                pass
            else:
                try:
                    idx = split_line.index("total_loss:")
                    total_loss.append(split_line[idx+1])
                except Exception:
                    pass

    table = zip(iteration, total_loss)
    print(len(iteration))
    print(len(total_loss))
    draw(table)


if __name__ == "__main__":
    main()
