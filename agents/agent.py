import math

import torch
import config
from torch import nn
import numpy as np

from config import train_stop
from utilities.replay_memory import Transition
import config

from config import Placement_LR, EPS_END, EPS_START, Placement_EPS_DECAY, NUM_HIDDEN_NEURON

class Agent():
    def __init__(self):
        self.device = None
        self.target_model = None
        self.policy_model = None
        self.memory = None
        self.optimizer = None
        self.loss = None
        self.steps_done = 0
        self.history = [0]*12

    def run(self, context):
        pass

    def reset_instance_metrics(self, context):
        #context.metrics.insert_instance(context.simulator.now, len(context.environment.users),
        #                                context.environment.nodes)
        yield context.simulator.timeout(1)
        #print("Instance reset")
        total = 0


        for node in context.environment.nodes:
            for instance in node.instances:
                total += instance.success_request + instance.unhandled_request + instance.timeout
                instance.response_time = 0.0
                instance.success_request = 0.0
                instance.unhandled_request = 0.0
                instance.timeout = 0.0
                instance.random_drop = 0.0
                instance.request_cnt = 0
                instance.request_record.clear()

        if total > 0:
            #print(f'{context.simulator.now} {total}')

            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12 =context.num_instances, context.total_response_time / context.total_received, context.total_sent, context.total_received, context.total_success, context.total_handled, context.total_unhandled, context.total_timeout, context.total_random_drop, context.total_network_drop, context.total_processing_drop, context.total_cloud_visits

            context.metrics.insert_eval_per_time(context.simulator.now,a1, a2, a3-self.history[2], a4-self.history[3], a5-self.history[4], a6-self.history[5], a7-self.history[6], a8-self.history[7], a9-self.history[8], a10-self.history[9], a11-self.history[10], a12-self.history[11])
            self.history =[context.num_instances, context.total_response_time / context.total_received, context.total_sent, context.total_received, context.total_success, context.total_handled, context.total_unhandled, context.total_timeout, context.total_random_drop, context.total_network_drop, context.total_processing_drop, context.total_cloud_visits]


    def learn(self):
        batch = self.memory.get()

        if len(batch) < config.BATCH_SIZE:
           return None

        batch = self.memory.sample(config.BATCH_SIZE)
        states, actions, next_states, rewards = zip(*batch)

        states = torch.cat(states).to(device=self.device)
        actions = torch.cat(actions).to(device=self.device)
        rewards = torch.cat(rewards).to(device=self.device)
        next_states = torch.cat(next_states).to(device=self.device)

        current_q = self.policy_model(states).gather(1, actions)

        max_next_q = self.target_model(next_states).max(1)[0]
        expected_q = rewards + (config.GAMMA * max_next_q)

        #criterion = nn.SmoothL1Loss()
        #loss = criterion(current_q, expected_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss = self.loss(current_q, expected_q.unsqueeze(1))
        loss.backward()
        #torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        # self.memory.clear()

        #
        # if self.episode % 50 == 0:
        #     self.LR *= 0.9
        #     self.optimizer = optim.Adam(self.model.parameters(), self.LR)

        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.policy_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * config.TAU + target_net_state_dict[key] * (1 - config.TAU)
        self.target_model.load_state_dict(target_net_state_dict)

        return loss.item() / len(batch)

    def get_probability_matrix(self, arr):
        softmax = nn.Softmax(dim=0)
        softmax_a = softmax(arr)
        return np.array(softmax_a.cpu())
