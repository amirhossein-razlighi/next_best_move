"""Task "pathfinding.terrain": grid maps with floor / mud / sand cells whose entry costs 1 / 2 / 4 are never shown; the target is
a cheapest route.

Splits: train, val, test (Table 2 "cheapest route") and probe (Table 2 "detour or cross": 250 maps where every cheapest route
crosses a band of costly terrain, 250 where every cheapest route goes around it). See nbm/tasks/pathfinding/.
"""
from nbm.tasks.pathfinding.task import Pathfinding


class Terrain(Pathfinding):
    terrain = True
    split_names = ("train", "val", "test", "probe")


TASK = Terrain
