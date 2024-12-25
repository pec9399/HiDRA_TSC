import numpy as np
import simpy

import config
from agents.agent import Agent
from models import PlacementRequest


class QT_Scaling(Agent):
    def __init__(self, node, seed, is_elder, placement_algorithm, file_path=None):
        self.node = node
        self.seed = seed
        self.file_path = file_path
        self.child_instances = []
        self.visited_cloud = 0
        self.episode = 1
        self.can_forward = True
        self.total_requests = 0
        self.total_handled = 0
        self.total_unhandled = 0
        self.total_timeout = 0
        self.total_random_drop = 0
        self.is_elder = True
        self.total_response = 0
        self.threshold = 30
        self.history = [0] * 12
        self.placement_algorithm = placement_algorithm()

    def run(self, context):
        try:
            while not context.stop:
                yield context.simulator.timeout(1000)  # 1초마다 행동

                msg = None
                # 다른 agent가 메시지 보냈음
                if len(context.agent_pipes[self.node.id].items) > 0:
                    msg = context.agent_pipes[self.node.id].items[0]
                    context.agent_pipes[self.node.id].items.pop(0)

                total_response = 0.0
                total_requests = 0
                total_unhandled = 0
                total_handled = 0
                total_timeout = 0
                total_random_drop = 0
                total_instance_cnt = 0
                total_requests_record = []
                finished = []
                if len(self.node.instances) > 0 or len(self.child_instances) > 0:
                    for instance in self.node.instances + self.child_instances:
                        if instance.id not in finished:
                            total_handled += instance.success_request  # 성공
                            total_unhandled += instance.unhandled_request  # drop (not enough instance)
                            total_timeout += instance.timeout  # 그냥 오래걸림

                            total_requests += instance.unhandled_request + instance.success_request + instance.timeout
                            total_response += instance.response_time
                            total_random_drop += instance.random_drop
                            total_instance_cnt += instance.request_cnt
                            total_requests_record += instance.request_record
                            finished.append(instance.id)


                    if total_requests > 0:
                        assert (total_response / total_requests <= 3000.0)
                        self.total_handled += total_handled
                        self.total_unhandled += total_unhandled
                        self.total_timeout += total_timeout
                        self.total_requests += total_requests
                        self.total_response += total_response
                        self.total_random_drop += total_random_drop

                        s = len(self.child_instances)
                        u = (total_unhandled+total_timeout) / len(self.child_instances)
                        p = 0.99
                        t = np.percentile(total_response, [p*100])
                        t = t[0]
                        qps = (s*u)+(np.log(1-p)/t)
                        #print(qps)
                        if qps > self.threshold:
                            self.placement_algorithm.next(context, self.node, msg)

                context.simulator.process(self.reset_instance_metrics(context))

        except simpy.Interrupt as i:
            context.logger.debug(f'Process Interrupt')