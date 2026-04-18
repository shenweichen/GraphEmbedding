import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ge.alias import alias_sample, create_alias_table


def gen_prob_dist(size):
    probabilities = np.random.randint(0, 100, size)
    return probabilities / np.sum(probabilities)


def simulate(size=100, sample_count=10000):
    truth = gen_prob_dist(size)
    accept, alias = create_alias_table(truth)

    sampled = np.zeros(size)
    for _ in range(sample_count):
        sampled[alias_sample(accept, alias)] += 1
    return sampled / np.sum(sampled), truth


def main(smoke=False, show=True):
    size = 20 if smoke else 100
    sample_count = 300 if smoke else 10000
    alias_result, truth = simulate(size=size, sample_count=sample_count)

    assert np.isclose(alias_result.sum(), 1.0)
    assert np.isclose(truth.sum(), 1.0)

    if show:
        plt.bar(list(range(len(alias_result))), alias_result, label="alias_result")
        plt.bar(list(range(len(truth))), truth, label="truth")
        plt.legend()
        plt.show()

    return alias_result, truth


if __name__ == "__main__":
    main()
