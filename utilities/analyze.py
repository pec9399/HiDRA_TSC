import os

import pandas as pd
from matplotlib import pyplot as plt

import config


def graph_loss(paths, is_elder, episode=None):
    for path in paths:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        path1 = f"../result/{path}/loss.csv"
        path2 = f"../result/{path}/rewards.csv"

        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)
        df1 = df1[df1['is_elder'] == is_elder]
        df2 = df2[df2['is_elder'] == is_elder]
        if episode is not None:
            df1 = df1[df1['episode'] <= episode]
            df2 = df2[df2['episode'] <= episode]

        p1 = df1.groupby('episode').agg({'loss': 'mean'})
        p2 = df2.groupby('episode').agg({'reward': 'mean'})
        ax1.plot(p1['loss'])
        ax2.plot(p2['reward'])
        plt.show()
        plt.close()


def graph_per_type(paths):
    for path in paths:
        fig, (ax) = plt.subplots(1, 1)
        path1 = f"../result/{path}/eval.csv"
        df1 = pd.read_csv(path1)


        ax.plot(df1.unhandled, label='Network queue overflow')
        ax.plot(df1.network_drop, label='latency ')
        ax.plot(df1.processing_drop, label='processing overflow')
        ax.plot(df1.random_drop, label='stochastic drop')
        #ax.plot(df1.random_drop + df1.network_drop, label='placement reward')

        plt.legend()
        plt.show()
        plt.close()

def graph_together(paths):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for path in paths:
        path_full = f"../result/{path}/eval.csv"
        df = pd.read_csv(path_full)

        ax1.plot(df['success'] * 100 / (df['total_received']), label=path)
        ax2.plot(df['num_instance'],label=path)



    plt.legend()
    plt.show()
    plt.close()

def tryMe():
    fig, ax = plt.subplots(1, 1)
    path1 = '2024-09-30-12-47-40-1/HiDRA'
    path = f"../result/{path1}/rewards.csv"
    df = pd.read_csv(path)
    df = df.groupby('episode').agg({'exploration': 'mean'})
    ax.plot(df.exploration)
    plt.show()
    plt.close()

def evaluate(paths):
    dfs = []
    fig, (ax) = plt.subplots(1, 1)
    fig, ax2 = plt.subplots(1, 1)
    fig, ax3 = plt.subplots(1, 1)
    fig, ax4 = plt.subplots(1, 1)
    ax.set_ylim([0, 100])
    ax2.set_ylim([0, 4000])
    ax3.set_ylim([0, 100])
    ax4.set_ylim([0, 100])
    names = ["HiDRA","HiDRA_NRS", "QoS-RR", "QT-RR", "CPU-RR", "QoS-L", "QT-L", "CPU-L"]
    for i, path in enumerate(paths):
        path_full = f"../result/{path}/eval.csv"
        path_time_full = f"../result/{path}/eval_per_time.csv"

        df = pd.read_csv(path_full)
        df2 = pd.read_csv(path_time_full)
        df2['rsr'] = df2['success'] * 100 / (df2['total_received'])
        df2_a = df2[df2['time'] > 150000]
        df2_b = df2[df2['time'] <= 150000]
        df2_a = df2_a.groupby('episode').mean()
        df2_b = df2_b.groupby('episode').mean()
        df = df[df['episode'] > config.train_stop]
        df['rsr'] = df['success']*100 / (df['total_received'])

        ax.bar(names[i], df['rsr'].mean(), width=0.5)
        ax.annotate(round(df['rsr'].mean(), 1), xy=(i-0.25, 80))
        ax.set_title("Response Success Rate")
        ax2.bar(names[i], df['num_instance'].mean(), width=0.5)
        ax2.annotate(round(df['num_instance'].mean()), xy=(i - 0.3,3500))
        ax2.set_title("Number of Instance")
        ax3.bar(names[i], df2_b['rsr'].mean(), width=0.5)
        ax3.annotate(round(df2_b['rsr'].mean(), 1), xy=(i-0.25, 85))
        ax3.set_title("Reward Success Rate (first half)")
        ax4.bar(names[i], df2_a['rsr'].mean(), width=0.5)
        ax4.annotate(round(df2_a['rsr'].mean(), 1), xy=(i - 0.25, 85))
        ax4.set_title("Reward Success Rate (second half)")
        print(df['rsr'].mean())
        print(df['num_instance'].mean())
        print(df2['rsr'].mean())
        print()
    plt.show()
    plt.close()

def evaluate_after_training(path):
    fig, ax1 = plt.subplots(1, 1)
    fig, ax2 = plt.subplots(1, 1)
    fig, ax3 = plt.subplots(1, 1)
    fig, ax4 = plt.subplots(1, 1)
    ax1.set_ylim([0,100])
    ax2.set_ylim([0,3000])
    ax1.set_xlim([50,config.train_stop+50])
    ax2.set_xlim([50,config.train_stop+50])

    ax1.set_xticks( range(100,config.train_stop+1,100))
    ax2.set_xticks( range(100,config.train_stop+1,100))

    ax3.set_xticks( range(100,config.train_stop+1,100))
    ax3.set_ylim([0, 100])
    ax3.set_xlim([50, config.train_stop+50])
    ax4.set_xticks(range(100, config.train_stop + 1, 100))
    ax4.set_ylim([0, 100])
    ax4.set_xlim([50, config.train_stop + 50])
    names = ["HiDRA", "QoS-RR", "QT-RR", "CPU-RR", "QoS-L", "QT-L", "CPU-L"]

    for i in range(100,config.train_stop+1,100):
        path_full = f"../result/{path}-{i}/eval.csv"
        path_time_full = f"../result/{path}-{i}/eval_per_time.csv"
        df = pd.read_csv(path_full)
        df2 = pd.read_csv(path_time_full)

        df2['rsr'] = df2['success'] * 100 / (df2['total_received'])
        df2_a = df2[df2['time'] > 150000]
        df2_b = df2[df2['time'] <= 150000]
        df2_a = df2_a.groupby('episode').mean()
        df2_b = df2_b.groupby('episode').mean()
        df['rsr'] = df['success'] * 100 / (df['total_received'])
        ax1.bar(i, df['rsr'].mean(), width=50)
        ax1.annotate(round(df['rsr'].mean(),1), xy=(i - 20, 70))
        ax1.set_title("Response Success Rate")
        ax2.bar(i, df['num_instance'].mean(), width=50)
        ax2.annotate(round(df['num_instance'].mean()), xy=(i-20,2500))
        ax2.set_title("Number of Instance")
        ax3.bar(i, df2_b['rsr'].mean(), width=50)
        ax3.annotate(round(df2_b['rsr'].mean(), 1), xy=(i-20, 70))
        ax3.set_title("Reward Success Rate (first half)")
        ax4.bar(i, df2_a['rsr'].mean(), width=50)
        ax4.annotate(round(df2_a['rsr'].mean(), 1), xy=(i-20, 85))
        ax4.set_title("Reward Success Rate (second half)")

        ax1.set_xlabel("Episode")
        ax2.set_xlabel("Episode")
        ax3.set_xlabel("Episode")
        ax4.set_xlabel("Episode")


        print(df['rsr'].mean())
        print(df['num_instance'].mean())
    plt.show()
    plt.close()

def print_per_time(paths):
    dfs = []
    fig, ax = plt.subplots(1, 1)

    fig,ax2 = plt.subplots(1, 1)

    for idx, path in enumerate(paths):
        path_time_full = f"../result/{path}/eval_per_time.csv"
        df2 = pd.read_csv(path_time_full)
        df2['rsr'] = df2['success'] * 100 / (df2['total_received'])
        #df2 = df2[df2['episode'] == episode]
        df2 = df2.groupby('time').mean()
        ax.plot(range(1,300,1), df2['rsr'], label=path.split('/')[1])
        ax.set_xlabel("Timestep (s)")
        ax.set_title("Response Success Rate")

        ax2.plot(range(1,300,1),df2['num_instance'], label=path.split('/')[1])
        ax2.set_xlabel("Timestep (s)")
        ax2.set_title("Number of Instance")


        ax.legend()
        ax2.legend()

    plt.legend()
    plt.show()
    plt.close()

def compare_convergence(paths, episode):
    fig, (ax1) = plt.subplots(1, 1)
    fig, ax2 = plt.subplots(1, 1)
    labels = ["Reward-sharing", "No reward-sharing"]
    for idx, path in enumerate(paths):

        path1 = f"../result/{path}/loss.csv"
        path2 = f"../result/{path}/rewards.csv"

        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)

        if episode is not None:
            df1 = df1[df1['episode'] <= episode]
            df2 = df2[df2['episode'] <= episode]

        p1 = df1.groupby('episode').agg({'loss': 'mean'})
        p2 = df2.groupby('episode').agg({'reward': 'mean'})
        ax1.plot(p1['loss'], label=labels[idx])
        ax2.plot(p2['reward'], label=labels[idx])
        ax1.legend()
        ax2.legend()
    plt.show()
    plt.close()


def final_graph():
    path = 'C:/Users/EunChan/Desktop/TSC_final result/results'

    baseline_names = ['QT-RR', 'QoS-RR', 'CPU-RR', 'QT-Latency', 'QoS-Latency', 'CPU-Latency']
    baseline_dfs = []
    baseline_dfs2 = []

    nors_names = ['HiDRA-500']
    nors_dfs = []
    nors_dfs2 = []

    for num_init_nodes in [20, 15, 10]:

        fig, ax1 = plt.subplots(1, 1)
        fig, ax2 = plt.subplots(1, 1)
        fig, ax3 = plt.subplots(1, 1)
        fig, ax4 = plt.subplots(1, 1)

        for i in range(1, 21, 1):
            path_baselines = f'{path}/baselines/{num_init_nodes}/{i}'
            path_nrs = f'{path}/no-reward-sharing/{num_init_nodes}/{i}'
            path_rs = f'{path}/reward-sharing/{num_init_nodes}/{i}'

            if os.path.exists(path_baselines):
                for idx, baseline_name in enumerate(baseline_names):
                    if len(baseline_dfs) < idx+1:
                        baseline_dfs.append(pd.read_csv(path_baselines + f'/{baseline_name}/eval.csv'))
                        baseline_dfs2.append(pd.read_csv(path_baselines + f'/{baseline_name}/eval_per_time.csv'))
                    else:
                        a = pd.read_csv(path_baselines + f'/{baseline_name}/eval.csv')
                        b = pd.read_csv(path_baselines + f'/{baseline_name}/eval_per_time.csv')
                        baseline_dfs[idx] = pd.concat([baseline_dfs[idx], a], ignore_index=True)
                        baseline_dfs2[idx] = pd.concat([baseline_dfs2[idx], b], ignore_index=True)

            if os.path.exists(path_nrs):
                for idx, nors_name in enumerate(nors_names):
                    if len(nors_dfs) < idx+1:
                        nors_dfs.append(pd.read_csv(path_nrs + f'/{nors_name}/eval.csv'))
                        nors_dfs2.append(pd.read_csv(path_nrs + f'/{nors_name}/eval_per_time.csv'))
                    else:
                        a = pd.read_csv(path_nrs + f'/{nors_name}/eval.csv')
                        b = pd.read_csv(path_nrs + f'/{nors_name}/eval_per_time.csv')
                        nors_dfs[idx] = pd.concat([nors_dfs[idx], a], ignore_index=True)
                        nors_dfs2[idx] = pd.concat([nors_dfs2[idx], b], ignore_index=True)

        for i, path in enumerate(nors_dfs):
            df = nors_dfs[i]
            df2 = nors_dfs2[i]
            df2['rsr'] = df2['success'] * 100 / (df2['total_received'])
            df2_a = df2[df2['time'] > 150000]
            df2_b = df2[df2['time'] <= 150000]
            df2_a = df2_a.groupby('episode').mean()
            df2_b = df2_b.groupby('episode').mean()
            df = df[df['episode'] > config.train_stop]
            df['rsr'] = df['success'] * 100 / (df['total_received'])

            ax1.bar(f'{nors_names[i]}_noRS', df['rsr'].mean(), width=0.5)
            ax2.bar(f'{nors_names[i]}_noRS', df2_b['rsr'].mean(), width=0.5)
            ax3.bar(f'{nors_names[i]}_noRS', df2_a['rsr'].mean(), width=0.5)
            ax4.bar(f'{nors_names[i]}_noRS', df['num_instance'].mean(), width=0.5)

        for i, path in enumerate(baseline_dfs):
            df = baseline_dfs[i]
            df2 = baseline_dfs2[i]
            df2['rsr'] = df2['success'] * 100 / (df2['total_received'])
            df2_a = df2[df2['time'] > 150000]
            df2_b = df2[df2['time'] <= 150000]
            df2_a = df2_a.groupby('episode').mean()
            df2_b = df2_b.groupby('episode').mean()
            df = df[df['episode'] > config.train_stop]
            df['rsr'] = df['success'] * 100 / (df['total_received'])

            ax1.bar(baseline_names[i], df['rsr'].mean(), width=0.5)
            ax2.bar(baseline_names[i], df2_b['rsr'].mean(), width=0.5)
            ax3.bar(baseline_names[i], df2_a['rsr'].mean(), width=0.5)
            ax4.bar(baseline_names[i], df['num_instance'].mean(), width=0.5)

        ax1.set_title(f"Response Success Rate - {num_init_nodes}")
        ax2.set_title(f"Reward Success Rate (first half) - {num_init_nodes}")
        ax3.set_title(f"Reward Success Rate (second half) - {num_init_nodes}")
        ax4.set_title(f"Number of Instance")
        plt.show()
        plt.close()
if __name__ == '__main__':
    final_graph()


