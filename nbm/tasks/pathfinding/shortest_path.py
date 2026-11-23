"""Task "pathfinding.shortest_path": uniform-cost grid maps (every step costs 1), the target is a shortest route.

Splits: train, val, test_id (alias "test"; Table 2 "held-out maps"), test_ood_long ("longer paths", 21+ steps), test_ood_size
("larger maps", 11 x 11 and 12 x 12) and test_ood_maze (the unseen maze family, appendix A.2). See nbm/tasks/pathfinding/.
"""
from nbm.tasks.pathfinding.task import Pathfinding


class ShortestPath(Pathfinding):
    terrain = False
    split_names = ("train", "val", "test_id", "test_ood_long", "test_ood_size", "test_ood_maze")
    aliases = {"test": "test_id"}


TASK = ShortestPath
