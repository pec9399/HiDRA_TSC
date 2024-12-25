from networkx import nodes

from agents.QT_scaling import QT_Scaling
from config import random

import pandas as pd

import config
from agents.CPU_scaling import CPU_Scaling
from agents.HiDRA_placement import HiDRA_Placement
from agents.HiDRA_scaling import HiDRA_Scaling
from agents.Latency_placement import Latency_Placement
from agents.QoS_scaling import QoS_Scaling
from agents.Roundrobin_placement import Roundrobin_Placement
from entities.node import Node
from entities.user import User
from environment import Environment
from models import Coordinate
from simulator import Simulator
from utilities.util import graph
from scipy.stats import qmc

from yafs import deterministicDistributionStartPoint


class Experiment:
    def __init__(self):
        pass

    def run(self, iteration_num, now, model_path=None, train_num = None):
        scaling_algorithms = [HiDRA_Scaling]  # , CPU_Scaling, CPU_Scaling]
        placement_algorithms = [HiDRA_Placement]  # , Roundrobin_Placement, Latency_Placement]
        algorithm_names = [f"HiDRA-{train_num}"]  # , "CPU-RR", "CPU-Latency"]
        # scaling_algorithms = [HiDRA_Scaling]
        # placement_algorithms = [HiDRA_Placement]
        # algorithm_names = ["HiDRA-500"]
        #scaling_algorithms = [QoS_Scaling, QT_Scaling, CPU_Scaling]
        #placement_algorithms = [Roundrobin_Placement, Roundrobin_Placement, Roundrobin_Placement]
        #algorithm_names = ["QoS-RR", "QT-RR", "CPU-RR"]
        #placement_algorithms = [Latency_Placement, Latency_Placement, Latency_Placement]
        #algorithm_names = ["QoS-Latency", "QT-Latency", "CPU-Latency"]

        # scaling_algorithms = [HiDRA_Scaling]*5  # , CPU_Scaling, CPU_Scaling]
        # placement_algorithms = [HiDRA_Placement]*5  # , Roundrobin_Placement, Latency_Placement]
        # algorithm_names = ["HiDRA-200", "HiDRA-400", "HiDRA-600",  "HiDRA-800", "HiDRA-1000"]

        environments = [
            Environment(seed=iteration_num, filePath=f"{now}-{iteration_num}/{algorithm}_{config.random_drop}/") for
            algorithm in algorithm_names]

        self.create_nodes(environments)
        self.create_users(environments)
        # self.create_random_nodes(environments)
        # self.create_random_users(environments)

        for j in range(0, len(environments)):
            elders = [n.id for n in environments[j].initial_nodes]
            for node in environments[j].nodes:
                if "HiDRA" in algorithm_names[j]:
                    model_num = None
                    if model_path is not None:
                        model_num = int(algorithm_names[j].split('-')[1])

                    if node.id in elders:
                        node.agent = scaling_algorithms[j](node, environments[j].seed, environments[j].filePath,
                                                           model_path, model_num)
                    else:
                        node.agent = placement_algorithms[j](node, environments[j].seed, environments[j].filePath,
                                                             model_path, model_num)
                else:
                    if node.id in elders:
                        node.agent = scaling_algorithms[j](node, environments[j].seed, node.id in elders,
                                                           placement_algorithms[j], environments[j].filePath)
                    node.placement_algorithm = placement_algorithms[j]()
        simulators = [
            Simulator(environment) for environment in environments
        ]

        for episode in range(config.episode_start, config.num_episodes + 1):
            for i in range(len(simulators)):
                print(algorithm_names[i])
                simulators[i].run(seed=iteration_num, episode=episode)
            if episode % 10 == 0:
                graph(algorithm_names, environments, episode)

    def create_random_nodes(self, environments):

        sampler = qmc.Halton(d=2, scramble=False)

        sample = sampler.random(n=config.num_nodes + 1)
        sample = sample[1:]
        for i, point in enumerate(sample):
            latency = random.randrange(10, 50)
            x = int(point[0] * config.map_width)
            y = int(point[1] * config.map_height)
            coverage = random.randrange(100, 150)
            # 0.1 = 90%
            reliability = 1.0  # use max to equally compare with no randomness
            reliability_degree = random.randrange(50, 2000) * 100
            c = Coordinate(x, y)
            possible_host = random.randrange(20, 40)
            for env in environments:
                node = Node(coordinate=c, id=i + 1, cpu_total=config.pod_CPU * possible_host, cpu_used=0,
                            memory_total=config.pod_memory * possible_host, memory_used=0,
                            coverage=coverage, store_total=config.pod_storage * possible_host, store_used=0,
                            latency=latency, reliability_prob=reliability, reliability_degree=reliability_degree)

                env.nodes.append(node)

        sampler = qmc.Halton(d=2, scramble=False)

        sample = sampler.random(n=config.random_drop + 1)
        random_nodes = []
        random_node_objects = []
        sample = sample[1:]
        for point in sample:
            x = int(point[0] * config.map_width)
            y = int(point[1] * config.map_height)
            c = Coordinate(x, y)
            min_node = -1
            min_dist = 100000
            for i in range(len(environments[0].nodes)):
                node = environments[0].nodes[i]
                dist = node.coordinate.distance(c)
                if dist < min_dist and environments[0].nodes[i].id not in random_nodes:
                    isolated = True
                    for nearby_node in random_node_objects:
                        d = nearby_node.coordinate.distance(node.coordinate)
                        if d < node.coverage or d < nearby_node.coverage:
                            isolated = False
                    if isolated:
                        min_dist = dist
                        min_node = i

            random_nodes.append(environments[0].nodes[min_node].id)
            random_node_objects.append(environments[0].nodes[min_node])
            # random_nodes.append(environments[0].nodes[random.randrange(0,len(environments[0].nodes))].id)
        sample = sampler.random(n=config.initial_instance + 1)
        initial_nodes = []
        sample = sample[1:]
        initial_node_objects = []
        for point in sample:
            if len(initial_nodes) < config.initial_instance:
                x = int(point[0] * config.map_width)
                y = int(point[1] * config.map_height)
                c = Coordinate(x, y)
                min_node = -1
                min_dist = 100000
                for i in range(len(environments[0].nodes)):
                    node = environments[0].nodes[i]
                    dist = node.coordinate.distance(c)
                    if dist < min_dist and environments[0].nodes[i].id not in random_nodes and environments[0].nodes[
                        i].id not in initial_nodes:
                        isolated = True
                        for nearby_node in initial_node_objects:
                            d = nearby_node.coordinate.distance(node.coordinate)
                            if d < node.coverage or d < nearby_node.coverage:
                                isolated = False
                        if isolated:
                            min_dist = dist
                            min_node = i
                initial_nodes.append(environments[0].nodes[min_node].id)
                initial_node_objects.append(environments[0].nodes[min_node])

        probabilities = [min(random.random(), 0.1)] * config.random_drop

        for env in environments:
            idx = 0
            for node in env.nodes:
                if node.id in random_nodes:
                    node.reliability_prob = probabilities[idx]
                    idx += 1
                if node.id in initial_nodes:
                    node.reliability_prob = 1.0
                    env.initial_nodes.append(node)
                if node.id <= 1:
                    continue
                for node2 in env.nodes:
                    if node2.id == node.id or node2.id <= 1:
                        continue
                    dist = node.distance(node2)
                    if dist <= node.coverage:
                        node.neighbors.append(node2)

    def create_nodes(self, environments):
        nodes_df = pd.read_csv('./data/nodes_cbd.csv')
        nodes_df = nodes_df[["LATITUDE", "LONGITUDE", "SITE_PRECISION"]]

        def convert(s):
            if s == 'Unknown':
                return 10
            else:
                tmp = s.split(' ')
                return int(tmp[1])

        nodes_precision = nodes_df[["SITE_PRECISION"]]
        nodes_df = (nodes_df[["LATITUDE", "LONGITUDE"]] - nodes_df[["LATITUDE", "LONGITUDE"]].min()) / (
                nodes_df[["LATITUDE", "LONGITUDE"]].max() - nodes_df[["LATITUDE", "LONGITUDE"]].min())
        nodes_df['SITE_PRECISION'] = nodes_precision['SITE_PRECISION'].map(convert)
        nodes_df_list = nodes_df.values.tolist()
        nodes_df_list = random.sample(nodes_df_list, config.num_nodes)

        for i in range(1, len(nodes_df_list) + 1):
            latency = random.randrange(10, 50)
            x = int(nodes_df_list[i - 1][1] * config.map_width)
            y = int(nodes_df_list[i - 1][0] * config.map_height)
            coverage = random.randrange(100, 150)
            # 0.1 = 90%
            reliability = 1.0  # use max to equally compare with no randomness
            reliability_degree = random.randrange(50, 2000) * 100
            c = Coordinate(x, y)
            possible_host = random.randrange(20, 40)
            for env in environments:
                node = Node(coordinate=c, id=i, cpu_total=config.pod_CPU * possible_host, cpu_used=0,
                            memory_total=config.pod_memory * possible_host, memory_used=0,
                            coverage=coverage, store_total=config.pod_storage * possible_host, store_used=0,
                            latency=latency, reliability_prob=reliability, reliability_degree=reliability_degree)

                env.nodes.append(node)

        sampler = qmc.Halton(d=2, scramble=False)
        sample = sampler.random(n=config.initial_instance + 1)
        initial_nodes = []
        sample = sample[1:]
        initial_node_objects = []
        for point in sample:
            if len(initial_nodes) < config.initial_instance:
                x = int(point[0] * config.map_width)
                y = int(point[1] * config.map_height)
                c = Coordinate(x, y)
                min_node = -1
                min_dist = 100000
                for i in range(len(environments[0].nodes)):
                    node = environments[0].nodes[i]
                    dist = node.coordinate.distance(c)
                    if dist < min_dist and environments[0].nodes[
                        i].id not in initial_nodes:
                        isolated = True
                        for nearby_node in initial_node_objects:
                            d = nearby_node.coordinate.distance(node.coordinate)
                            if d < node.coverage or d < nearby_node.coverage:
                                isolated = False
                        if isolated:
                            min_dist = dist
                            min_node = i
                initial_nodes.append(environments[0].nodes[min_node].id)
                initial_node_objects.append(environments[0].nodes[min_node])


        random_nodes = []
        random_node_objects = []
        for k in range(config.random_drop):
            x = random.randint(0,config.map_width)
            y = random.randint(0,config.map_height)
            c = Coordinate(x, y)
            min_node = -1
            min_dist = 100000
            for i in range(len(environments[0].nodes)):
                node = environments[0].nodes[i]
                dist = node.coordinate.distance(c)
                if dist < min_dist and environments[0].nodes[i].id not in random_nodes and environments[0].nodes[i].id not in initial_nodes:
                    isolated = True
                    for nearby_node in random_node_objects:
                        d = nearby_node.coordinate.distance(node.coordinate)
                        if d < node.coverage or d < nearby_node.coverage:
                            isolated = False
                    if isolated:
                        min_dist = dist
                        min_node = i

            random_nodes.append(environments[0].nodes[min_node].id)
            random_node_objects.append(environments[0].nodes[min_node])
            # random_nodes.append(environments[0].nodes[random.randrange(0,len(environments[0].nodes))].id)

        probabilities = [min(random.random(), 0.1)] * config.random_drop

        for env in environments:
            idx = 0
            for node in env.nodes:
                if node.id in random_nodes:
                    node.reliability_prob = probabilities[idx]
                    idx += 1
                if node.id in initial_nodes:
                    node.reliability_prob = 1.0
                    env.initial_nodes.append(node)
                if node.id <= 1:
                    continue
                for node2 in env.nodes:
                    if node2.id == node.id or node2.id <= 1:
                        continue
                    dist = node.distance(node2)
                    if dist <= node.coverage:
                        node.neighbors.append(node2)

    def create_random_users(self, environments):

        num_users = config.num_users
        num_nodes = len(environments[0].nodes)

        sampler = qmc.Halton(d=2, scramble=False)

        sample = sampler.random(n=config.num_users + 1)
        sample = sample[1:]

        for i in range(num_nodes + 1, num_users + num_nodes + 1):

            random_start = random.randint(0, 10000)
            random_interval = random.randint(100, 200)

            for env in environments:
                c = Coordinate(int(random.randrange(0, config.map_width)),
                               int(random.randrange(0, config.map_height)))

                d = User(c, i)
                d.distribution = deterministicDistributionStartPoint(random_start, random_interval)
                d.closestNode = env.nodes[0]

                minDist = d.distance(d.closestNode)
                for node in env.nodes:
                    if d.distance(node) < minDist:
                        minDist = d.distance(node)
                        d.closestNode = node
                env.users.append(d)

        for env in environments:
            for node in env.nodes:
                node.num_users = 0
                for user in env.users:
                    if node.distance(user) <= node.coverage:
                        node.num_users += 1

    def create_users(self, environments):
        user_df = pd.read_csv('./data/users.csv')
        user_df = (user_df - user_df.min()) / (user_df.max() - user_df.min())

        user_df_list = user_df.values.tolist()

        user_df_list = random.sample(user_df_list, config.num_users)

        # user_df = random.sample(user_df, config.num_users)
        # user_df = user_df.sample(config.num_users, ignore_index=True)
        num_users = config.num_users
        num_nodes = len(environments[0].nodes)

        for i in range(num_nodes + 1, num_users + num_nodes + 1):

            random_start = random.randint(0, 10000)
            random_interval = random.randint(100, 200)

            for env in environments:
                c = Coordinate(int(user_df_list[i - num_nodes - 1][1] * config.map_width),
                               int(user_df_list[i - num_nodes - 1][0] * config.map_height))

                d = User(c, i)
                d.distribution = deterministicDistributionStartPoint(random_start, random_interval)
                d.closestNode = env.nodes[0]

                minDist = d.distance(d.closestNode)
                for node in env.nodes:
                    if d.distance(node) < minDist:
                        minDist = d.distance(node)
                        d.closestNode = node
                env.users.append(d)

        for env in environments:
            for node in env.nodes:
                node.num_users = 0
                for user in env.users:
                    if node.distance(user) <= node.coverage:
                        node.num_users += 1
