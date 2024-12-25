import math
from datetime import datetime

import simpy
import torch
from matplotlib.style.core import available
from torch import nn, optim
import numpy as np
import config
from agents.agent import Agent
from config import Placement_LR, EPS_END, EPS_START, Placement_EPS_DECAY, NUM_HIDDEN_NEURON
from models import PlacementRequest
from utilities.DQN import DQN
from utilities.replay_memory import ReplayMemory
import random

class HiDRA_Placement(Agent):
    def __init__(self, node, seed, file_path=None, model_path = None, model_num = None):

        self.n_actions = 1 + 1 + 1 + len(node.neighbors)
        self.n_states = 5 + 1 + len(node.neighbors)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.target_model = DQN(self.n_states, self.n_actions).to(self.device)
        self.policy_model = DQN(self.n_states, self.n_actions).to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        # self.loss = nn.MSELoss()
        self.loss = nn.HuberLoss()
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=Placement_LR)

        self.steps_done = 0
        self.memory = ReplayMemory(6000)
        self.prev_state = None
        self.prev_path = None
        self.prev_action = None
        self.node = node

        self.child_instances = []
        self.reward = 0.0
        self.past_response_times = []
        self.visited_cloud = 0
        self.target = 3000.0
        self.episode = 1
        self.can_forward = True
        self.LR = Placement_LR
        self.total_requests = 0
        self.total_handled = 0
        self.total_unhandled = 0
        self.total_timeout = 0
        self.total_random_drop = 0
        self.is_elder = False
        self.total_response = 0
        self.total_cnt = 0
        self.seed = seed
        # debug
        self.explorations = 0
        self.exploitations = 0
        self.reward_records = []
        self.file_path = file_path
        self.prev_reward = 0
        self.model_path = model_path
        self.history = [0]*12
        #self.inference_times = []


        if model_path is not None:
            self.policy_model = torch.load(
                f'{model_path}/{self.node.id}-{model_num}')
            self.policy_model.to(self.device)
            print(f'Model: {model_path}/{self.node.id}-{model_num}')
        #assert(self.device != 'cuda:0')

    def memorize(self, state, action, next_state, reward, td_error=None):  # 에피소드 저장 함수
        self.memory.push(state,
                            action,
                            next_state,
                            torch.FloatTensor([reward]).to(device=self.device)
                         )

    def step(self, node, actions, time, path):
        assert (not self.is_elder)

        if path is not None:
            if self.prev_path is None:
                self.prev_path = path

        if self.prev_path is None:
            return 0, 'idle'


        now = datetime.now()
        total_response = 0.0
        total_requests = 0
        total_unhandled = 0
        total_handled = 0
        total_timeout = 0
        total_random_drop = 0

        cnt = 0
        if len(self.child_instances) > 0:
            for instance in self.child_instances:
                total_handled += instance.success_request  # 성공
                total_unhandled += instance.unhandled_request  # drop (not enough instance)
                total_timeout += instance.timeout  # 그냥 오래걸림

                total_requests += instance.unhandled_request + instance.success_request + instance.timeout
                total_response += instance.response_time
                total_random_drop += instance.random_drop
            # for instance in self.node.instances:
            #     instance.response_time = 0.0
            #     instance.success_request = 0
            #     instance.unhandled_request = 0
            #     instance.timeout = 0
            #     instance.random_drop = 0
            #     instance.request_cnt = 1
        possible_actions = actions
        if 0 not in actions:  # is forward
            unvisited_neighbors = list(range(3, self.n_actions))
            ids = [node.id for node in path]
            for i in range(len(node.neighbors)):
                neighbor = node.neighbors[i]
                if neighbor.id in ids or neighbor.agent.is_elder:
                    unvisited_neighbors.remove(3 + i)
            possible_actions = actions + unvisited_neighbors

        if len(possible_actions) == 0:
            self.can_forward = False
            stop_time = datetime.now() - now
            #self.inference_times.append(stop_time.total_seconds())
            return 0, 'No available action'

        my_node_timeout = 0
        my_node_random_drop = 0
        if len(self.node.instances) > 0:
            for instance in self.node.instances:
                my_node_timeout += instance.network_drop
                my_node_random_drop += instance.random_drop

        state_arr = list()
        state_arr.append(node.num_users)
        state_arr.append(len(node.instances))
        state_arr.append(total_timeout)
        state_arr.append(total_random_drop)
        state_arr.append(total_handled)
        state_arr.append(my_node_timeout)
        for n in node.neighbors:
            state_arr.append(n.agent.prev_reward)
        state = torch.FloatTensor([state_arr]).to(device=self.device)

        action, r = self.act(state, possible_actions)
        stop_time = datetime.now() - now
        #self.inference_times.append(stop_time.total_seconds())
        self.reward += -(total_random_drop)
        self.reward += -(total_timeout)
        #self.reward += total_handled
        if self.prev_action is None or total_requests == 0:
            self.prev_state = state
            self.prev_action = action
        else:
            #print(f'{total_requests} {total_handled} {total_unhandled} {total_timeout}')
            #self.reward += total_handled * 100 / total_requests

            # self.reward += total_handled
            self.prev_reward = self.reward
            self.reward_records.append(self.reward)
            self.memorize(self.prev_state, self.prev_action, state, self.reward)
            self.prev_state = state
            self.prev_action = action
            self.reward = 0.0
        return action.item(), r

    def penalize(self, reward):
        self.reward += -reward


    def place(self, context, msg):
        actions = [0]
        if msg is not None:  # elder 가 아닌데 forward를 받았다
            context.logger.debug(
                f'Episode {context.episode} at {context.simulator.now}: agent {self.node.id} - received forward and not elder')

            actions = [1]  # place on self 가능
            if not self.node.has_enough_resources():
                actions = []

        action_id, action_type = self.step(self.node, actions, context.simulator.now,
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
            assert (msg is not None)
            # place on self
            context.logger.debug(
                f'Episode {context.episode} at {context.simulator.now}: agent {self.node.id} - place on self')
            msg.path = msg.path + [self.node]
            context.node_pipes[self.node.id].put(msg)
        else:
            assert (action_id != 2 and msg is not None)
            # forward to neighbor

            node_id = self.node.neighbors[action_id - 3].id
            context.logger.debug(
                f'Episode {context.episode} at {context.simulator.now}: agent {self.node.id} - forward to {node_id} - {action_type}')
            destDevice = context.environment.findNodeById(node_id)

            p = PlacementRequest(-1, self.node, destDevice, 1, context.simulator.now)
            p.path = msg.path + [self.node]
            p.forward = True

            if type(destDevice.agent) == HiDRA_Placement:
                context.logger.debug(
                    f'Episode {context.episode} at {context.simulator.now}: agent {self.node.id} - forward to placement agent at {node_id} - {action_type}')
                destDevice.agent.place(context, p)
            else:
                context.logger.debug(
                    f'Episode {context.episode} at {context.simulator.now}: agent {self.node.id} - place on scaling agent {node_id} - {action_type}')
                context.agent_pipes[destDevice.id].put(p)
    def decide(self, context, msg=None):
        actions = [0]
        if msg is not None:  # elder 가 아닌데 forward를 받았다
            context.logger.debug(
                f'Episode {context.episode} at {context.simulator.now}: agent {self.node.id} - received forward and not elder')

            actions = [1]  # place on self 가능
            if not self.node.has_enough_resources():
                actions = []

        action_id, action_type = self.step(self.node, actions, context.simulator.now,
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
            assert (msg is not None)
            # place on self
            context.logger.debug(
                f'Episode {context.episode} at {context.simulator.now}: agent {self.node.id} - place on self')
            msg.path = msg.path + [self.node]
            context.node_pipes[self.node.id].put(msg)
        else:
            assert (action_id != 2 and msg is not None)
            # forward to neighbor

            node_id = self.node.neighbors[action_id - 3].id

            destDevice = context.environment.findNodeById(node_id)

            p = PlacementRequest(-1, self.node, destDevice, 1, context.simulator.now)
            p.path = msg.path + [self.node]
            p.forward = True

            if type(destDevice.agent) == HiDRA_Placement:
                context.logger.debug(
                    f'Episode {context.episode} at {context.simulator.now}: agent {self.node.id} - forward to placement agent at {node_id} - {action_type}')
                destDevice.place(context, p)
            else:
                context.logger.debug(
                    f'Episode {context.episode} at {context.simulator.now}: agent {self.node.id} - place on scaling agent {node_id} - {action_type}')
                context.agent_pipes[destDevice.id].put(p)
    def run(self, context):
        try:
            while not context.stop:
                yield context.simulator.timeout(1000)  # 1초마다 행동

                msg = None
                # 다른 agent가 메시지 보냈음
                if len(context.agent_pipes[self.node.id].items) > 0:
                    for item in context.agent_pipes[self.node.id].items:
                        msg = context.agent_pipes[self.node.id].items[0]
                        context.agent_pipes[self.node.id].items.pop(0)
                        self.decide(context,msg)
                else:
                    self.decide(context)

                context.simulator.process(self.reset_instance_metrics(context))

        except simpy.Interrupt as i:
            context.logger.debug(f'Process Interrupt')


    def act(self, state, possible_actions, nostep=False):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / config.Placement_EPS_DECAY)

        if not nostep:
            self.steps_done += 1

        if config.random.random() >= eps_threshold or nostep or self.episode > config.train_stop or self.model_path is not None:
            if self.episode == config.train_stop + 1 and self.model_path is None:
                #self.policy_model = torch.load(f'./model/{config.model_path}/HiDRA/{self.seed}/{self.node.id}-{config.train_stop}')
                self.policy_model = torch.load(f'./model/{self.file_path}{self.seed}/{self.node.id}-{config.train_stop}')
                self.policy_model.to(self.device)

            t = self.policy_model(state).data[0].clone().detach()


            for i in range(t.shape[0], -1, -1):
                if i not in possible_actions:
                    t = torch.cat([t[0:i], t[i+1:]])
            if not nostep:
                self.exploitations += 1

            # return argmax q among possible actions

            p = self.get_probability_matrix(t)

            if len(possible_actions) == 0:
                return torch.LongTensor([[0]]).to(device=self.device), "No available action"
            elif len(possible_actions) == 1:
                torch.LongTensor([[possible_actions[0]]]).to(device=self.device), "Inference"

            index = np.random.choice(possible_actions, p=p)

            return torch.LongTensor([[index]]).to(device=self.device), "Inference"

        else:
            assert (self.episode <= config.train_stop)

            rand_action = config.random.sample(possible_actions, 1)[0]
            if not nostep:
                self.explorations += 1

            return torch.LongTensor([[rand_action]]).to(device=self.device), "Random"
