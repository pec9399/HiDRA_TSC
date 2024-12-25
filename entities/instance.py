import simpy

import config
from simulator import Simulator
from config import random

class Instance():
    def __init__(self, app, id: int, cpu: int, memory: int, storage: int, node):
        self.app = app
        self.id = id
        self.node = node
        self.cpu = cpu
        self.memory = memory
        self.storage = storage

        self.path = []
        self.response_time = 0.0
        self.success_request = 0
        self.unhandled_request = 0
        self.timeout = 0
        self.random_drop = 0
        self.network_drop = 0
        self.processing_drop = 0
        self.request_cnt = 1
        self.request_record = []

    def run(self, context: Simulator):
        context.instance_pipes[self.id] = simpy.Store(context.simulator)
        try:
            while not context.stop:
                msg = yield context.instance_pipes[self.id].get()
                if not context.stop:
                    context.logger.debug(f'Node {self.node.id} instance {self.id} received message')
                    srcUser = msg.srcDevice
                    #yield context.simulator.timeout(msg.cpu / self.cpu * 1000)
                    msg.processing_delay = (msg.cpu / self.cpu * 1000) * self.request_cnt
                    self.request_cnt += 1
                    context.logger.debug(
                        f'Request {msg.id} Node processed request in {msg.processing_delay} / total - {msg.network_delay + msg.processing_delay}')
                    r = random.random()
                    msg.probabilistic_delay = 0
                    if r > self.node.reliability_prob:
                        context.logger.debug(
                            f'Request {msg.id} Request randomly dropped')
                        msg.probabilistic_delay = 3001.0#context.environment.random.randrange(0, self.node.reliability_degree)
                    msg.srcDevice = self

                    return_delay = self.node.distance(msg.destDevice)
                    if self.node.distance(msg.destDevice) <= self.node.coverage:
                        return_delay = self.node.latency

                    msg.network_delay += return_delay
                    context.logger.debug(
                        f'Request {msg.id} Node returning request / total - {msg.network_delay + msg.processing_delay}')
                    total_time = msg.network_delay + msg.processing_delay + msg.probabilistic_delay
                    msg.response_time = min(total_time, config.timeout)

                    context.total_response_time += min(total_time, config.timeout)
                    self.request_record.append(min(msg.network_delay + msg.processing_delay, config.timeout))
                    if total_time >= config.timeout:

                        context.logger.debug(
                            f'Request {msg.id} Request timeout Network: {msg.network_delay} / Processing: {msg.processing_delay} / Random : {msg.probabilistic_delay}')
                        context.total_timeout += 1
                        self.timeout += 1
                        if msg.probabilistic_delay > 0:
                            context.total_random_drop += 1
                            self.random_drop += 1
                        elif msg.network_delay >= config.timeout:
                            #if context.simulator.now > 1000*60*4:
                            #    context.logger.info(f'Episode {context.episode}, {context.simulator.now} - Network drop {srcUser.node.coordinate.x},{srcUser.node.coordinate.y} to Node at {self.node.coordinate.x}, {self.node.coordinate.y}')
                            self.network_drop += 1
                            context.total_network_drop += 1
                        elif msg.processing_delay >= config.timeout:
                            self.processing_drop += 1
                            context.total_processing_drop += 1
                        self.response_time += config.timeout
                    else:
                        context.logger.debug(
                            f'Request {msg.id} Success Network: {msg.network_delay} / Processing: {msg.processing_delay} / Random : {msg.probabilistic_delay}')
                        self.response_time += total_time
                        context.total_success += 1
                        self.success_request += 1

                    # self.metrics.insert(msg)
        except simpy.Interrupt as i:
            context.logger.debug(f'Process {self.id} interrupt')

    def run_overflow_handler(self, context: Simulator):
        context.instance_pipes[f'overflow-{self.id}'] = simpy.Store(context.simulator)
        try:
            while not context.stop:
                msg = yield context.instance_pipes[f'overflow-{self.id}'].get()
                if not context.stop:
                    msg.response_time = config.timeout
                    self.response_time += config.timeout
                    self.unhandled_request += 1
                    context.total_response_time += config.timeout
                    # self.metrics.insert(msg)

        except simpy.Interrupt as i:
            context.logger.debug(f'Process  interrupt')