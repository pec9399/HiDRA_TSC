
import math
from datetime import datetime

import numpy as np
import simpy
import torch
import torch.nn as nn
import torch.optim as optim

from agents.HiDRA_placement import HiDRA_Placement
from agents.agent import Agent
from config import EPS_END, EPS_START, EPS_DECAY, GAMMA, LR, NUM_HIDDEN_NEURON
from models import PlacementRequest
from utilities.DQN import DQN

from utilities.replay_memory import ReplayMemory

import config
import random

class HiDRA_Scaling(Agent):
    def __init__(self, node, seed, file_path=None, model_path = None, model_num = None):
        #do nothing + place self + forward to neighbors + become proxy
        self.n_actions = 1 + 1 + 1 + len(node.neighbors)
        self.n_states = 7 + len(node.neighbors)

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(config.device)
        self.target_model = DQN(self.n_states, self.n_actions).to(self.device)
        self.policy_model = DQN(self.n_states, self.n_actions).to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        #self.loss = nn.MSELoss()
        self.loss = nn.HuberLoss()
        self.optimizer = optim.Adam(self.policy_model.parameters(), LR)
        self.seed = seed
        self.steps_done = 1
        self.memory = ReplayMemory(6000)
        self.prev_state = None
        self.prev_path = None
        self.prev_action = None
        self.node = node

        self.child_instances = []
        self.reward = 0.0
        self.prev_reward = 0.0
        self.past_response_times = []
        self.visited_cloud = 0
        self.target = 3000.0
        self.episode = 1
        self.can_forward = True
        self.LR = LR
        self.total_requests = 0
        self.total_handled = 0
        self.total_unhandled = 0
        self.total_timeout = 0
        self.total_random_drop = 0
        self.is_elder = True
        self.total_response = 0
        self.random = random

        #debug
        self.explorations = 0
        self.exploitations = 0
        self.reward_records = []
        self.file_path = file_path
        self.model_path = model_path
        self.history = [0]*12
        #self.inference_times = []

        if model_path is not None:
            self.policy_model = torch.load(
                f'{model_path}/{self.node.id}-{model_num}')
            self.policy_model.to(self.device)
            print( f'Model: {model_path}/{self.node.id}-{model_num}')
        #assert(self.device != 'cuda:0')

    def memorize(self, state, action, next_state, reward, td_error=None):  # 에피소드 저장 함수
        self.memory.push( state,
                            action,
                            next_state,
                            torch.FloatTensor([reward]).to(device=self.device)
                         )

    def step(self, node, actions, time, path=None, forward=False, pipe=None):
        assert(self.is_elder)
        total_response = 0.0
        total_requests = 0
        total_unhandled = 0
        total_handled = 0
        total_timeout = 0
        total_random_drop = 0

        cnt = 0
        assert(len(self.child_instances) > 0)

        for n in self.node.instances:
            assert(n.id in [x.id for x in self.child_instances])
        now = datetime.now()

        if len(self.child_instances) > 0:

            for instance in self.child_instances:
                total_handled += instance.success_request #성공
                total_unhandled += instance.unhandled_request #drop (not enough instance)
                total_timeout += instance.timeout #그냥 오래걸림

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
            if total_requests > 0:
                assert(total_response / total_requests <= 3000.0)
                self.total_handled += total_handled
                self.total_unhandled += total_unhandled
                self.total_timeout += total_timeout
                self.total_requests += total_requests
                self.total_response += total_response
                self.total_random_drop += total_random_drop
        #queue with capacity=5
        if total_response != 0.0:
            self.past_response_times.append(total_unhandled)
            if len(self.past_response_times) > 5:
                self.past_response_times.pop(0)
        else:
            stop_time = datetime.now() - now
            #self.inference_times.append(stop_time.total_seconds())
            return 0, 'no reward collected'

        # if len(self.past_response_times) < 5:
        #     return 0, 'Not enough state'

        #self.reward += -abs(self.target - total_response)

        #self.reward += -(self.total_unhandled * self.target * 5)
        #self.reward += -(self.total_timeout * self.target) #네트워크 + 프로세싱
        #self.reward += (self.total_handled / self.total_requests) * self.target
        self.reward += -(total_unhandled)

        #self.reward += -(total_timeout)
        #self.reward += total_handled / total_requests * 100

        #self.reward += total_handled
        self.reward += -(len(self.child_instances))


        #
        instance_handle_ps = config.pod_CPU // config.request_CPU
        cpu_average = (total_handled + total_timeout) / (instance_handle_ps * len(self.child_instances))

        state_arr = []

        state_arr.append(cpu_average)
        state_arr.append(len(self.child_instances))
        state_arr.append(node.num_users)
        state_arr.append(total_requests)
        state_arr.append(total_timeout)
        state_arr.append(total_random_drop)
        state_arr.append(total_handled)
        prev_reward_total = 0
        cnt = 0
        for n in node.neighbors:

            state_arr.append(n.agent.prev_reward)
            if not n.agent.is_elder:
                prev_reward_total += n.agent.prev_reward
                cnt += 1
        if cnt > 0:
            self.reward += prev_reward_total / cnt

        self.prev_reward = self.reward




        assert (len(state_arr) == self.n_states)
        state = torch.FloatTensor([state_arr]).to(device=self.device)
        unvisited_neighbors =list(range(3,self.n_actions))

        if path is not None:
            ids = [n.id for n in path]
            for i in range(len(node.neighbors)):
                neighbor = node.neighbors[i]
                if neighbor.id in ids:
                    unvisited_neighbors.remove(3+i)

        for i in range(len(node.neighbors)):
            neighbor = node.neighbors[i]
            if neighbor.agent.is_elder:
                if 3+i in unvisited_neighbors:
                    unvisited_neighbors.remove(3 + i)

        if 1 not in actions and  len(unvisited_neighbors) == 0:
            assert(not self.node.has_enough_resources())
            for n in node.neighbors:
                if type(n.agent) == HiDRA_Placement:
                    if(not n.has_enough_resources()):
                        print(f"{time} {[x.id for x in path]}")
                        assert(False)
            actions.append(2)

        if forward:
            possible_actions = actions + unvisited_neighbors
        else:
            possible_actions = actions
        if len(possible_actions) == 0:
            self.can_forward = False
            stop_time = datetime.now() - now
            #self.inference_times.append(stop_time.total_seconds())
            return 0, 'No available action'
        #if total_unhandled == 0:
        #    possible_actions = [0]
        action, r = self.act(state, possible_actions)
        stop_time = datetime.now() - now
        #self.inference_times.append(stop_time.total_seconds())
        #first action
        if self.prev_state is None:
            self.prev_state = state
            self.prev_action = action
        else:
            assert(total_response != 0.0)
            self.reward_records.append(self.reward)
            self.memorize(self.prev_state, self.prev_action, state, self.reward)
            self.prev_state = state
            self.prev_action = action
            self.reward = 0.0

        return action.item(), r


    def penalize(self, reward):
        self.reward += -reward

    def decide(self, context, msg=None):

        actions = [0, 1]

        if not self.node.has_enough_resources():
            actions.remove(1)
        if msg is not None and msg.forward:
            actions.remove(0)

        # elder만 새로 생성
        elders = [n.id for n in context.environment.initial_nodes]
        forward = True

        if self.node.id not in elders and msg is None:
            forward = False

        # forward된 요청이 있으면 그 요청의 path까지 보냄 -> agent는 path에 있는 노드는 행동에서 거름
        action_id, type = self.step(self.node, actions, context.simulator.now,
                                    msg.path if msg is not None and msg.forward else None,
                                    forward=forward, pipe=context.instance_pipes)

        # 아무것도 안함
        if action_id == 0:
            context.logger.debug(
                f'Episode {self.episode} at {context.simulator.now}: agent {self.node.id} - do nothing - {type}')
            if msg is not None and msg.forward:
                context.logger.debug(
                    f'terminate forward request, msg path {[node.id for node in msg.path]} {type}')
                cnt = 1
                # for node_in_path in msg.path:
                #    node_in_path.agent.penalize(50 * cnt)
                #    cnt += 1

        # 자신한테 instance 추가
        elif action_id == 1:
            context.logger.debug(
                f'Episode {context.episode} at {context.simulator.now}: agent {self.node.id} - place on self - {type}')

            p = PlacementRequest(-1, self.node, self.node, 1, context.simulator.now)
            p.path = msg.path + [self.node] if msg is not None else [self.node]
            p.forward = False
            context.node_pipes[self.node.id].put(p)
        # cloud 한테 SOS (자기한테 리소스 없을때만 가능)
        elif action_id == 2:
            context.logger.debug(
                f'Episode {context.episode} at {context.simulator.now}: agent {self.node.id} - visit cloud - {type}')

            p = PlacementRequest(-1, self.node, self.node, 1, context.simulator.now)
            p.forward = True
            context.node_pipes[f'cloud'].put(p)
        # coverage 안에 있는 neighbor한테 forward
        else:
            node_id = self.node.neighbors[action_id - 3].id
            context.logger.debug(
                f'Episode {context.episode} at {context.simulator.now}: agent {self.node.id} - forward to {node_id} - {type}')
            destDevice = context.environment.findNodeById(node_id)

            if destDevice.has_enough_resources():
                p = PlacementRequest(-1, self.node, destDevice, 1, context.simulator.now)
                p.path = msg.path + [self.node] if msg is not None else [self.node]
                p.forward = True
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
                        self.decide(context, msg)
                else:
                    self.decide(context)
                context.simulator.process(self.reset_instance_metrics(context))

        except simpy.Interrupt as i:
            context.logger.debug(f'Process Interrupt')


    def act(self, state, possible_actions, nostep=False):

        eps_threshold = EPS_END + (EPS_START-EPS_END)* math.exp(-1. * self.steps_done / config.EPS_DECAY)

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
                    t = torch.cat([t[0:i], t[i + 1:]])
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
            assert(self.episode <= config.train_stop)

            rand_action = config.random.sample(possible_actions, 1)[0]
            if not nostep:
                self.explorations += 1

            return torch.LongTensor([[rand_action]]).to(device=self.device), "Random"
