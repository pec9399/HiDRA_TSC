import simpy

import config
from simulator import Simulator
from config import random

class Cloud():
    def __init__(self):
        pass

    def run(self, context: Simulator):
        try:
            while not context.stop:
                msg = yield context.node_pipes['cloud'].get()
                if not context.stop:
                    #global scheduler add instance on an available node
                    initial_nodes = context.environment.initial_nodes
                    candidates = []
                    for node in initial_nodes:
                        if node.has_enough_resources() and node.agent.can_forward:
                            candidates.append(node)
                    if len(candidates) > 0:
                        msg.destDevice = random.sample(candidates,1)[0]
                        msg.path = [msg.destDevice]
                        msg.forward = True
                        if msg.srcDevice.agent is not None:
                            msg.srcDevice.agent.visited_cloud += 1
                        context.total_cloud_visits += 1
                        context.agent_pipes[msg.destDevice.id].put(msg)
                    else:

                        candidates = []
                        for node in context.environment.nodes:
                            if node.has_enough_resources():
                                candidates.append(node)
                        assert(len(candidates) > 0)
                        msg.destDevice = random.sample(candidates, 1)[0]
                        msg.path = [msg.destDevice]
                        msg.forward = True
                        if msg.srcDevice.agent is not None:
                            msg.srcDevice.agent.visited_cloud += 1
                        context.total_cloud_visits += 1
                        context.agent_pipes[msg.destDevice.id].put(msg)



        except simpy.Interrupt as i:
            context.logger.debug(f'Process Cloud interrupt')

