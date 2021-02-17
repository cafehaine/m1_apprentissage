import random
from typing import Dict, Optional

from island import Island, Coordinates

EPSILON = 0.9


def epsilongreedy(
        island: Island, *, coords: Optional[Coordinates]=None, remaining_turns: int = 20, print_island: bool=False
) -> int:
    if coords is None:
        coords = island.random_coords()

    if print_island:
        print("-" * island._width * 2)
        island.print(coords)

    if remaining_turns <= 0:
        return 0

    action_kind = random.random()

    # Logical move
    if action_kind <= EPSILON:
        possibilities: Dict[Coordinates, int] = {}
        for action in island.possible_actions(coords):
            possibilities[action] = island.get_value(action)

        max_score = max(possibilities.values())
        top_actions = []
        for action, score in possibilities.items():
            if score == max_score:
                top_actions.append(action)

        new_coords = random.choice(top_actions)
    # Random move
    else:
        rand_coords = island.random_possible_action(coords)
        if rand_coords is None:
            raise RuntimeError("Could not get a random action.")
        new_coords = rand_coords

    output = island.take_contents(new_coords)
    return output + epsilongreedy(island, coords=new_coords, remaining_turns=remaining_turns - 1, print_island=print_island)
