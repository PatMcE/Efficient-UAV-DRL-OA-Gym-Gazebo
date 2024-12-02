'''
This code is from https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code
'''

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Average Reward vs Timesteps')
    plt.savefig(filename)
