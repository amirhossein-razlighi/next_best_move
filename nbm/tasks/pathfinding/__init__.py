"""Pathfinding on grid maps, two tasks sharing this package (paper Table 2, appendix A.2):

  pathfinding.shortest_path   uniform costs: reach the goal along a shortest route        (shortest_path.py)
  pathfinding.terrain         floor / mud / sand cost 1 / 2 / 4, never shown: a cheapest route   (terrain.py)

Modules: maps (cells, generators, BFS / Dijkstra targets, route checks), render (pictures, training targets, the cell decoder and
the threshold-free move readout), data (the data builder), evaluation (image closed loop, video clip execution, metrics), task
(the nbm Task shared by both).
"""
