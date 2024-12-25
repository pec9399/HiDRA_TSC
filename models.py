import numpy as np
class Coordinate:
    def __init__(self, x: float, y:float):
        self.x = x
        self.y = y

    def update(self, x: float, y: float):
        self.x = x
        self.y = y

    def vectorize(self):
        return [self.x, self.y]

    def tuple(self):
        return self.x, self.y

    def size(self) -> float:
        """ size: returns the size of the vector """
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def distance(self, other) -> float:
        """ get_distance: get distance between to vectors """
        assert isinstance(other, Coordinate)
        return (other - self).size()

    def __sub__(self, other):
        return Coordinate(self.x - other.x, self.y - other.y)
class Entity:
    def __init__(self, coordinate: Coordinate):
        self.coordinate = coordinate

    def distance(self, other) -> float:
        assert isinstance(other, Entity)
        return self.coordinate.distance(other.coordinate)

class Request():
    def __init__(self, id,srcDevice:Entity,destDevice: Entity,timestamp):
        self.id = id
        self.srcDevice = srcDevice
        self.destDevice = destDevice
        self.timestamp = timestamp

class UserRequest(Request):
    def __init__(self,id,srcDevice, destDevice, cpu: int, memory: int, bytes: int, timestamp):
        super().__init__(id,srcDevice, destDevice,timestamp)
        self.cpu = cpu
        self.memory = memory
        self.bytes = bytes
        self.response_time = -1
        self.network_delay = -1
        self.processing_delay = -1
        self.probabilistic_delay = -1
        self.global_registry = False

class PlacementRequest(Request):
    def __init__(self,id,srcDevice: Entity, destDevice: Entity, n: int,timestamp):
        super().__init__(id, srcDevice, destDevice, timestamp)
        self.n = n
        self.delay = 0
        self.initial = False
        self.path = []
        self.forward = False
