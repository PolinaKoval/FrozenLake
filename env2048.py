from grid import Grid
import random


class Vector(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Traversals(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        

class Game2048(object):
    
    def __init__(self, size):
        self.size = size
        self.start_tiles = 2
        self.action_space = 4
        self.observation_space = 8
        self.over = False
        self.won = False
        self.grid = Grid(self.size)
        self.score = 0

    def reset(self):
        self.over = False
        self.won = False
        self.grid = Grid(self.size)
        self.score = 0

        self._add_start_tiles()

        return self.grid

    def _add_start_tiles(self):
        for x in range(self.start_tiles):
            self._add_rendom_tile()

    def _add_rendom_tile(self):
        value = 2 if random.uniform(0, 1) < 0.9 else 4
        self.grid.insert_value_in_random_available_cell(value)

    def _get_vector(self, direction):
        return [Vector(0, -1), Vector(1, 0), Vector(0, 1), Vector(-1, 0)][direction]

    def _build_traversals(self ,vector):
        x = range(self.size)[::-1] if vector.x == 1 else range(self.size)
        y = range(self.size)[::-1] if vector.y == 1 else range(self.size)

        return Traversals(x, y)

    def _prepare_tiles(self):
        for row in self.grid:
            for cell in row:
                cell.merged = False

    def _find_farthest_position(self, cell, vector):
        previous = None
        while True:
            previous = cell
            cell = self._get_next_tile(cell, vector)
            if cell is None or cell.value != 0: 
                break

        return previous, cell

    def _get_next_tile(self, tile, vector):
        newX = tile.x + vector.x
        newY = tile.y + vector.y
        if self._is_in_bounds(Vector(newX, newY)):
            return self.grid[newX][newY]

        return None

    def _is_in_bounds(self, vector):
        return 0 <= vector.x < self.size and 0 <= vector.y < self.size

    def _moves_available(self):
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid[x][y]
                if tile.value == 0 or self._is_tile_matches_available(tile):
                    return True

    def _is_tile_matches_available(self, tile):
        for direction in range(self.action_space):
            vector = self._get_vector(direction)
            neighbor = self._get_next_tile(tile, vector)
            if neighbor is not None and neighbor.value == tile.value:
                return True

        return False

    def step(self, direction): 
        start_score = self.score
        vector = self._get_vector(direction)
        traversals = self._build_traversals(vector)
        moved = False

        self._prepare_tiles()

        for x in traversals.x:
            for y in traversals.y:
                tile = self.grid[x][y]
                if tile.value == 0:
                    continue
                farthest, next_cell = self._find_farthest_position(tile, vector)

                if next_cell is not None and next_cell.value == tile.value and not next_cell.merged:
                    new_value = tile.value * 2
                    next_cell.value = new_value
                    self.score += new_value
                    if new_value == 2048:
                        self.won = True
                    moved = True

                    next_cell.merged = True
                    tile.value = 0
                else:
                    if farthest.x != tile.x or farthest.y != tile.y:
                        farthest.value = tile.value
                        tile.value = 0
                        moved = True

        if moved:
            self._add_rendom_tile()

            if not self._moves_available():
                self.over = True

        return self.grid, self.score - start_score, self.won or self.over, moved
