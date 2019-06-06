import sys
import matplotlib.pyplot as plt

def main(fname):
    steps = []
    losses = []
    with open(fname, 'r') as f:
        for line in f:
            if "step" in line and "test" not in line:
                words = line.split()
                index = words.index("step")
                step = int(words[index+1][:-1])
                steps.append(step)
                loss = float(words[index+4])
                losses.append(loss)

    plt.plot(steps[::100], losses[::100])
    plt.title("Training loss of landmark detector")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()
    return steps, losses


if __name__ == "__main__":
    fname = sys.argv[1]
    main(fname)
