from agents.agent import Agent
from models import PlacementRequest
from simulator import Simulator


class Latency_Placement():
    def __init__(self):
        self.is_elder = False
        self.history = [0]*12

    def next(self, context:Simulator, node, msg):

        #place on self
        if node.has_enough_resources():
            p = PlacementRequest(-1, node, node, 1, context.simulator.now)
            if msg is not None:
                p.path = msg.path + [node]
            else:
                p.path = [node]
            p.forward = True
            context.node_pipes[node.id].put(p)
        else:
            unvisited_neighbors = list(range(len(node.neighbors)))
            if msg is not None and msg.forward:
               ids = [node.id for node in msg.path]
               for i in range(len(node.neighbors)):
                   neighbor = node.neighbors[i]
                   if neighbor.id in ids:
                       unvisited_neighbors.remove(i)

            #can't place on self, visited all the neighbors
            if len(unvisited_neighbors) == 0:
                p = PlacementRequest(-1, node, node, 1, context.simulator.now)
                p.forward = True
                context.node_pipes[f'cloud'].put(p)
            #place on nearest unvisited neighbor
            else:
                dest_neighbor = unvisited_neighbors[0]
                min_dist = node.distance(node.neighbors[dest_neighbor])
                for action in unvisited_neighbors:
                   neighbor = node.neighbors[action]
                   if node.distance(neighbor) < min_dist and neighbor.has_enough_resources():
                       dest_neighbor = action
                       min_dist = node.distance(neighbor)
                dest_device = node.neighbors[dest_neighbor]
                p = PlacementRequest(-1, node, dest_device, 1, context.simulator.now)
                if msg is not None:
                    p.path = msg.path + [node]
                else:
                    p.path = [node]
                p.forward = True
                context.node_pipes[dest_device.id].put(p)