import matplotlib.pyplot as plt
import numpy as np

from ge.alias import alias_sample, create_alias_table


def gen_prob_dist(N):
    p = np.random.randint(0, 100, N)
    return p/np.sum(p)


def simulate(N=100, k=10000,):

    truth = gen_prob_dist(N)

    area_ratio = truth
    accept, alias = create_alias_table(area_ratio)

    ans = np.zeros(N)
    for _ in range(k):
        i = alias_sample(accept, alias)
        ans[i] += 1
    return ans/np.sum(ans), truth


if __name__ == "__main__":
    alias_result, truth = simulate()
    plt.bar(list(range(len(alias_result))), alias_result, label='alias_result')
    plt.bar(list(range(len(truth))), truth, label='truth')
    plt.legend()
