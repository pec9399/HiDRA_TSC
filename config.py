import random as r

EPS_START = 0.99  # 학습 시작시 에이전트가 무작위로 행동할 확률
# ex) 0.5면 50% 절반의 확률로 무작위 행동, 나머지 절반은 학습된 방향으로 행동
# random하게 EPisolon을 두는 이유는 Agent가 가능한 모든 행동을 경험하기 위함.
EPS_END = 0.05   # 학습 막바지에 에이전트가 무작위로 행동할 확률
#EPS_START에서 END까지 점진적으로 감소시켜줌.
# --> 초반에는 경험을 많이 쌓게 하고, 점차 학습하면서 똑똑해지니깐 학습한대로 진행하게끔
EPS_DECAY = 5000  # 학습 진행시 에이전트가 무작위로 행동할 확률을 감소시키는 값, 높을수록 느림
GAMMA = 1.0    # 할인계수 : 에이전트가 현재 reward를 미래 reward보다 얼마나 더 가치있게 여기는지에 대한 값.
# 일종의 할인율
LR = 0.01     # 학습률
BATCH_SIZE = 128  # 배치 크기
TAU = 0.005

NUM_HIDDEN_NEURON = 16

Placement_EPS_DECAY = 5000
Placement_LR = 0.01


map_width = 1000
map_height = 1000

pod_CPU = 2000
pod_memory = 2000
pod_storage = 2000
request_CPU = 1000
request_memory = 1000
request_bytes = 1000


num_nodes = 124
num_users = 204

iterations = 20
episode_time = 5*60*1000
episode_start = 1
episode_end = 500
num_episodes = 500
initial_instance = 20
random_drop = 20
train_stop = 500
timeout = 3000.0

random = r

device = "cuda"

model_path = '2024-09-28-19-38-57-1'
