import simpy

import config
from models import Entity, Coordinate, UserRequest
from simulator import Simulator

from yafs import deterministicDistributionStartPoint
from config import random

class User(Entity):
    def __init__(self, coordinate: Coordinate, id):
        super().__init__(coordinate)
        self.closestNode = None
        self.id = id
        self.distribution = None

    def run(self, context:Simulator):
        try:
            yield context.simulator.timeout(1)  # 1ms 뒤 시작가능
            while not context.stop:
                nextTime = self.distribution.next()
                yield context.simulator.timeout(nextTime)
                if not context.stop:
                    closest_node = self.closestNode

                    initial_nodes = [n.id for n in context.environment.initial_nodes]

                    if closest_node.id in initial_nodes or len(closest_node.instances) > 0:
                        # network delay
                        if(len(closest_node.instances) == 0):
                            print(initial_nodes)
                            print(closest_node.id)
                        if closest_node.distance(self) < closest_node.coverage:
                            latency = closest_node.latency  # random between 10 and 50
                        else:
                            latency = closest_node.latency + closest_node.distance(self)*2


                        for i in range(1):
                            rid = context.get_request_id()

                            request = UserRequest(rid, self, closest_node, config.request_CPU, config.request_memory,
                                                  config.request_bytes, context.simulator.now)
                            request.network_delay += latency
                            context.env_processes.append(context.simulator.process(context.send(request, latency)))
                            context.total_sent += 1

                    else:
                        # query global service registry
                        serving_nodes = []
                        for node in context.environment.nodes:
                            if len(node.instances) > 0:
                                serving_nodes.append(node)
                        d = closest_node.distance(serving_nodes[0])
                        min_node = serving_nodes[0]  # 가장 가까운 elder
                        for serving_node in serving_nodes:
                            if closest_node.distance(serving_node) < d:
                                d = closest_node.distance(serving_node)
                                min_node = serving_node
                        for i in range(1):
                            rid = context.get_request_id()
                            request = UserRequest(rid, self, min_node, config.request_CPU, config.request_memory,
                                                  config.request_bytes, context.simulator.now)
                            request.global_registry = True
                            # 얼마만큼 줘야될까..?
                            overhead = len(context.environment.nodes)*20

                            # elder와의 거리 + query overhead
                            latency = closest_node.latency + self.distance(min_node)*2 + overhead
                            request.network_delay += latency

                            context.logger.debug(f'Request {rid} Access global registry and send to {request.destDevice.id} with latency {latency}')
                            context.env_processes.append(context.simulator.process(context.send(request, latency)))
                            context.total_sent += 1

        except simpy.Interrupt as i:
            context.logger.debug(f'Process interrupt')

    def reset(self):
        random_start = random.randint(0, 10000)
        random_interval = random.randint(30, 70)
        self.distribution = deterministicDistributionStartPoint(random_start, random_interval)