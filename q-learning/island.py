import random
from typing import List, NamedTuple, Optional


class Coordinates(NamedTuple):
    x: int
    y: int


class Island:
    def __init__(self, width: int, height: int, n_rhum: int) -> None:
        if width * height < n_rhum + 1:
            raise ValueError("Island too small to contain the rhum and the treasure.")

        self._width = width
        self._height = height

        self._board = [[0 for __ in range(width)] for _ in range(height)]

        # Place the rhum
        for _ in range(n_rhum):
            placed = False
            while not placed:
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                if self._board[y][x] == 0:
                    self._board[y][x] = 2
                    placed = True

        # Place the treasure
        placed = False
        while not placed:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if self._board[y][x] == 0:
                self._board[y][x] = 10
                placed = True

    def possible_actions(self, coords: Coordinates) -> List[Coordinates]:
        """Return all possible moves."""
        output = []
        if coords.x > 0:
            output.append(Coordinates(coords.x - 1, coords.y))
        if coords.x < self._width - 1:
            output.append(Coordinates(coords.x + 1, coords.y))
        if coords.y > 0:
            output.append(Coordinates(coords.x, coords.y - 1))
        if coords.y < self._height - 1:
            output.append(Coordinates(coords.x, coords.y + 1))
        return output

    def random_possible_action(self, coords: Coordinates) -> Optional[Coordinates]:
        """Return a random possible action."""
        possible_actions = self.possible_actions(coords)
        if not possible_actions:
            return None
        return random.choice(possible_actions)

    def take_contents(self, coords: Coordinates) -> int:
        """Take the value out of a cell."""
        output = self._board[coords.y][coords.x]
        self._board[coords.y][coords.x] = 0
        return output

    def get_value(self, coords: Coordinates) -> int:
        """Return the value of a cell."""
        return self._board[coords.y][coords.x]

    def random_coords(self) -> Coordinates:
        """Return some random coordinates."""
        return Coordinates(
            random.randint(0, self._width - 1), random.randint(0, self._height - 1)
        )
