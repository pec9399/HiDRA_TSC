import simpy

import config
from agents.HiDRA_placement import HiDRA_Placement
from agents.HiDRA_scaling import HiDRA_Scaling
from entities.instance import Instance
from models import Entity, Coordinate, UserRequest, PlacementRequest
from simulator import Simulator
from utilities.util import capture_screenshot


class Node(Entity):
    def __init__(self, coordinate: Coordinate, id: int, coverage: float, cpu_total: int, cpu_used: int,
                 memory_total: int, memory_used: int, store_total: int, store_used: int, latency: float, reliability_prob: float, reliability_degree: int):
        super().__init__(coordinate)
        self.coverage = coverage
        self.cpu_total = cpu_total
        self.cpu_used = cpu_used
        self.memory_total = memory_total
        self.memory_used = memory_used
        self.store_total = store_total
        self.store_used = store_used
        self.neighbors = []
        self.id = id
        self.instances = []
        self.latency = latency
        self.network_queue = None
        self.reliability_prob = reliability_prob
        self.reliability_degree = reliability_degree
        self.agent = None
        self.num_users = 0
        self.users = []

        self.placement_algorithm =None


    def has_enough_resources(self):
        return self.cpu_total - self.cpu_used >= config.pod_CPU and self.memory_total - self.memory_used >= config.pod_memory and self.store_total - self.memory_used >= config.pod_storage


    def run(self, context: Simulator):
        try:
            round_robin = 0

            while not context.stop:
                msg = yield context.node_pipes[self.id].get()
                if not context.stop:
                    if type(msg) == UserRequest:
                        # load balancing
                        context.request_per_timestep += 1

                        context.total_received += 1
                        #context.logger.info(f"Time: {context.simulator.now}, Total received: {context.total_received}")
                        msg.processing_delay = context.simulator.now
                        if self.agent is not None and self.agent.is_elder:

                            assert (len(self.agent.child_instances) > 0 or len(self.instances) > 0)
                            next_instance = self.agent.child_instances[round_robin]
                            if self.distance(next_instance.node) > self.coverage:
                                msg.network_delay += self.latency + self.distance(next_instance.node)*2
                            elif self.distance(next_instance.node) > 0:
                                msg.network_delay += self.latency
                            context.logger.debug(
                                f'Request {msg.id} Node received user request, time passed - {msg.network_delay}')
                            msg.processing_delay = context.simulator.now  # record time
                            #context.logger.info(f'{context.simulator.now} Node {self.id} ')
                            # if instance network queue is not full (1초당 처리 할 수 있는 요청의 3배)
                            if next_instance.request_cnt < (config.pod_CPU // msg.cpu)*3:
                                context.total_handled += 1
                                context.instance_pipes[next_instance.id].put(msg)
                            else:

                                context.logger.debug(f"Request {msg.id} Elder: not enough pod to handle request")
                                context.total_unhandled += 1
                                context.instance_pipes[f'overflow-{next_instance.id}'].put(msg)
                            round_robin += 1
                            round_robin %= len(self.agent.child_instances)
                        else:
                            if len(self.instances) == 0:
                                context.logger.debug(
                                    f'Request {msg.id} Node received user request without instance, time passed - {msg.network_delay}')
                            else:
                                context.logger.debug(
                                    f'Request {msg.id} Node received user request, time passed - {msg.network_delay}')
                            assert(len(self.instances) > 0)
                            next_instance = self.instances[round_robin]
                            round_robin += 1
                            round_robin %= len(self.instances)
                            msg.processing_delay = context.simulator.now
                            if next_instance.request_cnt < (config.pod_CPU // msg.cpu)*3:
                                context.total_handled += 1
                                context.instance_pipes[next_instance.id].put(msg)
                            else:
                                context.logger.debug(f"Request {msg.id} Normal: not enough pod to handle request")
                                context.total_unhandled += 1
                                context.instance_pipes[f'overflow-{next_instance.id}'].put(msg)
                    elif type(msg) == PlacementRequest:
                        # 지금은 n=1임
                        for i in range(msg.n):
                            context.logger.debug(f"at {context.simulator.now} Node {self.id} received placement request, path: [{[node.id for node in msg.path]}]")

                            if self.has_enough_resources():
                                mid = context.get_instance_id()
                                instance = Instance('SimpleApp', mid, config.pod_CPU, config.pod_memory,
                                                    config.pod_storage, self)
                                instance.path = msg.path
                                self.instances.append(instance)
                                self.cpu_used += instance.cpu
                                self.memory_used += instance.memory
                                self.store_used += instance.storage


                                for node in msg.path:
                                    if node.agent is not None and node.agent.is_elder:
                                        context.logger.debug(
                                            f"at {context.simulator.now} Child instance added to Node {node.id} (Scaling agent)")

                                        node.agent.child_instances.append(instance) #scaling agent
                                if msg.path[0].id != self.id:
                                    context.logger.debug(
                                        f"at {context.simulator.now} Child instance added to Node {node.id} (Last placement agent)")
                                    if self.agent is not None:
                                        self.agent.child_instances.append(instance) #placement agent

                                if len(msg.path) > 1:
                                    if msg.path[-2].id != self.id and msg.path[-2].id != msg.path[0].id:
                                        context.logger.debug(
                                            f"at {context.simulator.now} Child instance added to Node {node.id} (Second placement agent)")
                                        if msg.path[-2].agent is not None:
                                            msg.path[-2].agent.child_instances.append(instance) #previous placement agent

                                # instance process to handle userequests
                                #context.instance_pipes[mid] = simpy.Store(context.simulator)

                                # message queue for network overflow
                                #context.instance_pipes[f'overflow-{mid}'] = simpy.Store(context.simulator)

                                context.instance_processes.append(
                                    context.simulator.process(instance.run(context)))
                                context.env_processes.append(
                                    context.simulator.process(instance.run_overflow_handler(context)))
                                context.num_instances += 1
                                context.logger.debug(f"at {context.simulator.now} 1 instance placed on Node {self.id}")

                            else:
                                context.logger.debug(
                                    f"at {context.simulator.now} Not enough resource to place on Node {self.id}")
                                if msg.forward:
                                    if type(self.agent) == HiDRA_Placement:
                                        self.agent.place(context, msg)
                                    else:
                                        self.placement_algorithm.next(context,self, msg)
                            if context.episode == 501:
                                if context.simulator.now == 0:
                                    capture_screenshot(context.environment, context.episode,f'{context.simulator.now}')

        except simpy.Interrupt as i:
            context.logger.debug(f'Process interrupt')


    def place(self, context, msg):
        for i in range(msg.n):
            context.logger.debug(
                f"at {context.simulator.now} Node {self.id} received placement request, path: [{[node.id for node in msg.path]}]")

            if type(self.agent) == HiDRA_Placement:

                actions = [1]  # place on self 가능
                if not self.has_enough_resources():
                    actions = []

                action_id, action_type = self.agent.step(self, actions, context.simulator.now,
                                                                       msg.path if msg is not None else None)
                if action_id == 0:
                    if action_type == 'No available action':  # dead end
                        context.logger.debug(
                            f'terminate forward request, msg path {[node.id for node in msg.path]} {action_type}')
                        cnt = 1
                        # for node_in_path in msg.path:
                        #    node_in_path.agent.penalize(50 * cnt)
                        #    cnt += 1
                elif action_id == 1:
                    mid = context.get_instance_id()
                    instance = Instance('SimpleApp', mid, config.pod_CPU, config.pod_memory,
                                        config.pod_storage, self)
                    instance.path = msg.path
                    self.instances.append(instance)
                    self.cpu_used += instance.cpu
                    self.memory_used += instance.memory
                    self.store_used += instance.storage

                    self.agent.child_instances.append(instance)
                    if len(msg.path) > 1:
                        msg.path[-2].agent.child_instances.append(instance)

                    # instance process to handle userequests
                    # context.instance_pipes[mid] = simpy.Store(context.simulator)

                    # message queue for network overflow
                    # context.instance_pipes[f'overflow-{mid}'] = simpy.Store(context.simulator)

                    context.instance_processes.append(
                        context.simulator.process(instance.run(context)))
                    context.env_processes.append(
                        context.simulator.process(instance.run_overflow_handler(context)))
                    context.num_instances += 1
                    context.logger.debug(f"at {context.simulator.now} 1 instance placed on Node {self.id}")
                else:
                    node_id = self.neighbors[action_id - 3].id
                    context.logger.debug(
                        f'Episode {context.episode} at {context.simulator.now}: agent {self.id} - forward to {node_id} - {type}')
                    destDevice = context.environment.findNodeById(node_id)

                    p = PlacementRequest(-1, self, destDevice, 1, context.simulator.now)
                    p.path = msg.path + [self]
                    p.forward = True
                    if type(destDevice.agent) == HiDRA_Placement:
                        destDevice.place(context, p)
                    else:
                        context.agent_pipes[destDevice.id].put(p)

            else:
                if self.has_enough_resources():
                    mid = context.get_instance_id()
                    instance = Instance('SimpleApp', mid, config.pod_CPU, config.pod_memory,
                                        config.pod_storage, self)
                    instance.path = msg.path
                    self.instances.append(instance)
                    self.cpu_used += instance.cpu
                    self.memory_used += instance.memory
                    self.store_used += instance.storage

                    # for node_in_path in msg.path:
                    #     context.logger.debug(f'Child instance added to {node_in_path.id}')
                    #     node_in_path.agent.child_instances.append(instance)
                    self.agent.child_instances.append(instance)
                    if len(msg.path) > 1:
                        msg.path[-2].agent.child_instances.append(instance)

                    # instance process to handle userequests
                    # context.instance_pipes[mid] = simpy.Store(context.simulator)

                    # message queue for network overflow
                    # context.instance_pipes[f'overflow-{mid}'] = simpy.Store(context.simulator)

                    context.instance_processes.append(
                        context.simulator.process(instance.run(context)))
                    context.env_processes.append(
                        context.simulator.process(instance.run_overflow_handler(context)))
                    context.num_instances += 1
                    context.logger.debug(f"at {context.simulator.now} 1 instance placed on Node {self.id}")
                else:
                    context.logger.debug(
                        f"at {context.simulator.now} Not enough resource to place on Node {self.id}")
                    if msg.forward:
                        self.placement_algorithm.next(context, self, msg)