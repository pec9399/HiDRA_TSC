
import pandas as pd

import config
from entities.cloud import Cloud
from entities.node import Node
from entities.user import User
from models import Coordinate
from yafs import deterministicDistributionStartPoint


class Environment:
    def __init__(self, seed, filePath):
        self.seed = seed
        self.nodes = list[Node]()
        self.users = list[User]()
        self.initial_nodes = list[Node]()
        self.cloud = Cloud()
        self.filePath = filePath


    def findNodeById(self, id):
        for node in self.nodes:
            if node.id == id:
                return node
        return -1
