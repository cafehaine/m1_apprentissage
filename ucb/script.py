import math

from manchot import creer_manchots
from strategies import *


def main(n=15, iters=15000, k=math.sqrt(2)):
    set_k(k)
    machines = creer_manchots(n)
    score_random = recherche_aleatoire(iters, machines)
    print("Score random", score_random)
    score_best = recherche_gloutonne(iters, machines)
    print("Score Meilleur", score_best)
    score_ucb = recherche_ucb(iters, machines)
    print("Score UCB", score_ucb)


if __name__ == "__main__":
    main(k=10)
