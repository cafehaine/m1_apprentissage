from random import gauss, uniform
from typing import List


class Manchot:
    def __init__(self, esperance: float, variance: float) -> None:
        self.esperance = esperance
        self.variance = variance

    def tirer_bras(self) -> float:
        return gauss(self.esperance, self.variance)


def creer_manchots(nb: int) -> List[Manchot]:
    output = []
    for _ in range(nb):
        output.append(Manchot(uniform(-10, 10), uniform(0, 10)))
    return output
