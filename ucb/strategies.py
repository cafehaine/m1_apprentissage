import math
from random import choice
from typing import Sequence

from manchot import Manchot

K = math.sqrt(2)


def set_k(k: float) -> None:
    global K
    K = k


def recherche_aleatoire(nb_iter: int, machines: Sequence[Manchot]) -> float:
    total = 0
    for _ in range(nb_iter):
        machine = choice(machines)
        total += machine.tirer_bras()
    return total


def recherche_gloutonne(nb_iter: int, machines: Sequence[Manchot]) -> float:
    total = 0

    visited_all = False
    best_machine = machines[0]
    best_result = -math.inf

    for i in range(nb_iter):
        if visited_all:
            total += best_machine.tirer_bras()
        else:
            result = machines[i].tirer_bras()
            if result > best_result:
                best_result = result
                best_machine = machines[i]
            total += result
            if i == len(machines) - 1:
                visited_all = True

    return total


def recherche_ucb(nb_iter: int, machines: Sequence[Manchot]) -> float:
    total = 0

    tirages: Dict[Manchot, int] = {machine: 0 for machine in machines}
    scores: Dict[Manchot, float] = {machine: 0 for machine in machines}

    for i in range(nb_iter):
        # UCB
        if i >= len(machines):
            machine = machines[0]
            best_ucb = -math.inf
            for manchot in machines:
                ucb = (
                    scores[manchot] / tirages[manchot]
                    + K * math.log(i) / tirages[manchot]
                )
                if ucb > best_ucb:
                    best_ucb = ucb
                    machine = manchot
        # Init
        else:
            machine = machines[i]

        result = machine.tirer_bras()
        total += result
        tirages[machine] += 1
        scores[machine] += result

    print(
        "Répartition des essais UCB:", " ".join([str(val) for val in tirages.values()])
    )

    return total
