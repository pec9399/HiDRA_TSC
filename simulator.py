import gc
import logging
import os
from datetime import datetime

import simpy
import torch

import config
from agents.HiDRA_placement import HiDRA_Placement

from agents.HiDRA_scaling import HiDRA_Scaling
from models import PlacementRequest
from utilities.metrics import Metrics


class Simulator:
    def __init__(self, environment):
        self.environment = environment
        self.simulator = simpy.Environment()
        self.node_processes = []
        self.agent_processes = []
        self.user_processes = []
        self.env_processes = []
        self.instance_processes = []
        self.node_pipes = {}
        self.instance_pipes = {}
        self.agent_pipes = {}
        self.logger = logging.getLogger(__name__)
        self.request_id = 0
        self.instance_id = 0
        self.register_users()
        self.register_nodes()

        self.total_sent = 0.0
        self.total_received = 0.0
        self.total_handled = 0.0
        self.total_unhandled = 0.0
        self.total_timeout = 0.0
        self.total_random_drop = 0.0
        self.total_network_drop = 0.0
        self.total_processing_drop = 0.0
        self.total_success = 0.0
        self.num_instances = 0.0
        self.total_response_time = 0.0
        self.total_cloud_visits = 0
        self.stop = False
        self.metrics = None
        self.episode = 0
        self.request_per_timestep = 0


    def register_users(self):
        for user in self.environment.users:
            self.user_processes.append(self.simulator.process(user.run(self)))

    def register_nodes(self):
        for node in self.environment.nodes:
            self.node_pipes[node.id] = simpy.Store(self.simulator)
            self.agent_pipes[node.id] = simpy.Store(self.simulator)
            self.node_processes.append(self.simulator.process(node.run(self)))
            if node.agent is not None:
                if node.agent.is_elder or  type(node.agent) == HiDRA_Placement:
                    self.agent_processes.append(self.simulator.process(node.agent.run(self)))



        self.node_pipes['cloud'] = simpy.Store(self.simulator)
        self.node_processes.append(self.simulator.process(self.environment.cloud.run(self)))

    def send(self, request, delay:float, DESID=None):
        yield self.simulator.timeout(delay)
        if not self.stop:
            if DESID is None:
                self.node_pipes[request.destDevice.id].put(request)
            else:
                self.node_pipes[DESID].put(request)

    def get_request_id(self):
        self.request_id += 1
        return self.request_id

    def get_instance_id(self):
        self.instance_id += 1
        return self.instance_id

    def reset(self, episode):
        self.stop = True
        try:
            processes = self.node_processes + self.agent_processes, self.user_processes + self.env_processes + self.instance_processes
            for process in processes:
                if process.target != None:
                    process.interrupt('reset')
                    self.simulator.step()

        except:
            pass
        self.simulator = simpy.Environment()

        self.node_pipes = {}
        self.instance_pipes = {}
        self.agent_pipes = {}
        self.request_id = 0
        self.instance_id = 0
        self.node_processes = []
        self.agent_processes = []
        self.user_processes = []
        self.env_processes = []
        self.instance_processes = []


        self.total_sent = 0.0
        self.total_received = 0.0
        self.total_handled = 0.0
        self.total_unhandled = 0.0
        self.total_timeout = 0.0
        self.total_random_drop = 0.0
        self.total_network_drop = 0.0
        self.total_processing_drop = 0.0
        self.total_success = 0.0
        self.num_instances = 0.0
        self.total_response_time = 0.0
        self.total_cloud_visits = 0
        self.stop = False
        self.metrics = None
        self.episode = episode

        for node in self.environment.nodes:
            for instance in node.instances:
                del instance

            node.instances = []
            node.cpu_used = 0
            node.memory_used = 0
            node.store_used = 0
            if type(node.agent) == HiDRA_Placement or type(node.agent) == HiDRA_Scaling:
                node.agent.child_instances = []
                node.agent.reward_records = []
                node.agent.prev_state = None
                node.agent.prev_action = None
                node.agent.prev_path = None
                node.agent.visited_cloud = 0
                node.agent.total_handled = 0
                node.agent.can_forward = True
                node.agent.total_unhandled = 0
                node.agent.total_timeout = 0
                node.agent.total_requests = 0
                node.agent.total_random_drop = 0
                node.agent.total_response = 0
                node.agent.reward = 0.0
                node.agent.prev_reward = 0.0

                node.agent.total_handled = 0
                node.agent.episode = self.episode
                node.agent.explorations = 0
                node.agent.exploitations = 0
                node.agent.past_response_times = []
                #node.agent.steps_done = 0

            else:
                if node.agent is not None:
                    node.agent.total_response = 0
                    node.agent.child_instances = []
                    node.agent.visited_cloud = 0
                    node.agent.can_forward = True
                    node.agent.total_unhandled = 0
                    node.agent.total_timeout = 0
                    node.agent.total_requests = 0
                    node.agent.total_random_drop = 0
                    node.agent.total_handled = 0
                    node.agent.total_unhandled = 0
                    node.agent.episode = self.episode
        for user in self.environment.users:
            user.reset()

        gc.collect()
        self.stop = False


    def run(self,seed, episode):

        print(f"Starting episode {episode}")
        self.reset(episode)

        now = datetime.now()

        self.metrics = Metrics(self.environment.filePath, episode, seed)
        self.register_nodes()
        self.register_users()

        for node in self.environment.initial_nodes:
            r = PlacementRequest(0, node, node, 1, self.simulator.now)
            r.initial = True
            r.path.append(node)
            self.logger.debug(f'Sending initial placement request to {node.id}')
            self.node_pipes[node.id].put(r)

        self.simulator.run(until=config.episode_time + 1)
        stop_time = datetime.now() - now
        for node in self.environment.nodes:
            if type(node.agent) == HiDRA_Placement or type(node.agent) == HiDRA_Scaling:

                node.agent.episode = episode
                #self.metrics.insert_inference_time(episode, node.id, node.agent.inference_times)
                if episode <= config.train_stop:

                    loss = node.agent.learn()
                    if loss is not None:
                        self.metrics.insert_loss(episode, node, loss)


        if (episode % 100 == 0):
            for node in self.environment.nodes:
                if type(node.agent) == HiDRA_Placement or type(node.agent) == HiDRA_Scaling:
                    if not os.path.exists('./model'):
                        os.makedirs('./model')
                    if not os.path.exists(f'./model/{self.environment.filePath}'):
                        os.makedirs(f'./model/{self.environment.filePath}')
                    if not os.path.exists(f'./model/{self.environment.filePath}/{self.environment.seed}'):
                        os.makedirs(f'./model/{self.environment.filePath}/{self.environment.seed}')
                    torch.save(node.agent.target_model, f'./model/{self.environment.filePath}/{self.environment.seed}/{node.id}-{episode}')
        self.metrics.record_reward(self.environment.nodes)
        self.metrics.insert_eval(self.num_instances, self.total_response_time / self.total_received, self.total_sent, self.total_received, self.total_success, self.total_handled, self.total_unhandled, self.total_timeout, self.total_random_drop, self.total_network_drop, self.total_processing_drop, self.total_cloud_visits)
        self.metrics.close()

        self.stop = True

        print('Finished episode in ' + str(stop_time.total_seconds()))
        gc.collect()
