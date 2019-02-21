from matplotlib import pyplot as plt
import os
def plot_avg_reward(path):

    with open(path, "r") as f:
        data = f.readlines()
        epochs = []
        scores = []
        for line in data[1:]:
            epoch, score = line.strip().split(",")
            epochs.append(int(epoch)*100)
            scores.append(float(score))

    fig, ax = plt.subplots()
    ax.plot(epochs, scores)
    ax.axhline(13, color="red")
    ax.set_xlabel("number of episodes")
    ax.set_ylabel("avg score of 100 episodes")

    ax.set_title("Averge Score Plot for Banana Env")
    plt.savefig("score.png")

if __name__ == '__main__':
    path = "score.csv"
    plot_avg_reward(path)

