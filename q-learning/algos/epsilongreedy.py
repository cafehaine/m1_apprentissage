import random
from typing import Dict, Optional

from island import Island, Coordinates

EPSILON = 0.9


def epsilongreedy(
    island: Island, coords: Optional[Coordinates], remaining_turns: int = 20
) -> int:
    if remaining_turns <= 0:
        return 0

    if coords is None:
        coords = island.random_coords()

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
    return output + epsilongreedy(island, new_coords, remaining_turns - 1)
