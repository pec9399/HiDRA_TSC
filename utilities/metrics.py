import csv
import logging
import os

from agents.HiDRA_placement import HiDRA_Placement
from agents.HiDRA_scaling import HiDRA_Scaling
from models import UserRequest
import shutil

class Metrics:
    def __init__(self,filePath, episode, experiment):
        columns_event = ['episode','timestamp','src', 'dst','network_delay','processing_delay','probabilistic_delay','response_time']

        path = f"./result/{filePath}"

        if not os.path.exists(path):
            os.makedirs(path)

        self.path = path
        self.files = {}

        self.addFile('reward',f'{path}rewards.csv', ['episode','agent','is_elder','steps','created_instances','exploration','exploitation','reward'])
        #self.addFile('user', f'{path}user/{episode}-user_metrics.csv',['episode','timestamp','src', 'dst','network_delay','processing_delay','probabilistic_delay','response_time'])
        self.addFile('instance',f'{path}instance_metrics.csv', ['episode','time','num_users','received','handled','unhandled','total_response_time','total_instances'])
        self.addFile('loss',f'{path}loss.csv', ['episode', 'agent', 'is_elder','loss', 'num_child','child'])
        self.addFile('eval',f'{path}eval.csv', ['episode','num_instance','avg_response_time','total_received','total_sent','success','handled','unhandled', 'timeout','random_drop','network_drop', 'processing_drop', 'cloud_visits'])
        self.addFile('eval_per_time',f'{path}eval_per_time.csv', ['episode','time','num_instance','avg_response_time','total_received','total_sent','success','handled','unhandled', 'timeout','random_drop','network_drop', 'processing_drop', 'cloud_visits'])
        self.addFile('inference',f'{path}inference.csv', ['episode', 'node', 'time', 'inference_time'])
        self.episode = episode
        self.done = False
        self.logger = logging.getLogger(__name__)
        shutil.copyfile('./config.py', f'{path}config.py')


    def addFile(self, key, file_name, columns):
        if not os.path.exists(file_name):
            file = open(file_name, 'w')
            fw = csv.writer(file)
            fw.writerow(columns)
            self.files[key] = (fw, file)
        else:
            file = open(file_name, 'a')
            fw = csv.writer(file)
            self.files[key] = (fw, file)

    def insert(self,msg:UserRequest):
        rt = min(msg.network_delay + msg.probabilistic_delay + msg.processing_delay, 3000)

        self.files['user'][0].writerow([self.episode, msg.timestamp,msg.destDevice.id, msg.srcDevice.id, msg.network_delay, msg.processing_delay, msg.probabilistic_delay,  rt])

    def insert_loss(self, episode,node,loss):

        s = [instance.id for instance in node.agent.child_instances]
        self.files['loss'][0].writerow([episode,node.id,1 if node.agent.is_elder else 0,loss, len(node.agent.child_instances),f'{s}'])

    def insert_inference_time(self, episode, node, inference_times):
        for idx,time in enumerate(inference_times):
            self.files['inference'][0].writerow([episode,node,idx+1,time])

    def insert_instance(self, time, num_users, fog_nodes):
        handled = 0
        unhandled = 0
        total_instances = 0
        total_response_time = 0
        visited_cloud = 0
        random_drop = 0
        for node in fog_nodes:
            for instance in node.instances:
                total_instances += 1
                handled += instance.handled_request
                unhandled += instance.unhandled_request
                total_response_time += instance.response_time
                random_drop += instance.random_drop
            visited_cloud += node.agent.visited_cloud

        if handled + unhandled > 0:
            total_response_time /= (handled + unhandled)
        self.files['instance'][0].writerow([self.episode, time, num_users, handled+unhandled, handled, unhandled, total_response_time, total_instances, visited_cloud])
        self.done = True

    def record_reward(self, fog_nodes):
        for node in fog_nodes:
            if type(node.agent) == HiDRA_Scaling or type(node.agent) == HiDRA_Placement:
                for rec in node.agent.reward_records:
                    self.files['reward'][0].writerow([self.episode, node.id, 1 if node.agent.is_elder else 0, node.agent.steps_done, len(node.agent.child_instances), node.agent.explorations, node.agent.exploitations, rec])

    def insert_eval(self, num_instance,total_response,total_sent,total_received, total_success, total_handled, unhandled, timeout, random_drop,network_drop, processing_drop,cloud_visits):
        #self.logger.info(f'{self.episode}, {num_instance}, {total_response}, {total_received}, {total_success}, {total_handled}, {unhandled}, {timeout}, {random_drop}, {cloud_visits}')
        print(f'{self.episode}, {num_instance}, {total_response}, {total_sent}, {total_received}, {total_success}, {total_handled}, {unhandled}, {timeout}, {random_drop}, {network_drop}, {processing_drop}, {cloud_visits}')

        self.files['eval'][0].writerow([self.episode,num_instance, total_response,total_sent, total_received, total_success,total_handled, unhandled,timeout,random_drop,network_drop, processing_drop, cloud_visits])
    def insert_eval_per_time(self, time, num_instance,total_response,total_sent,total_received, total_success, total_handled, unhandled, timeout, random_drop,network_drop, processing_drop,cloud_visits):
        #self.logger.info(f'{self.episode}, {num_instance}, {total_response}, {total_received}, {total_success}, {total_handled}, {unhandled}, {timeout}, {random_drop}, {cloud_visits}')
        #print(f'{self.episode}, {num_instance}, {total_response}, {total_sent}, {total_received}, {total_success}, {total_handled}, {unhandled}, {timeout}, {random_drop}, {network_drop}, {processing_drop}, {cloud_visits}')

        self.files['eval_per_time'][0].writerow([self.episode,time, num_instance, total_response,total_sent, total_received, total_success,total_handled, unhandled,timeout,random_drop,network_drop, processing_drop, cloud_visits])
    def close(self):
        for file in self.files.items():
            file[1][1].close()
