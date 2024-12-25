import os

import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from scipy.stats import qmc

import config

def capture_screenshot(environment, episode, id):
    img = Image.new('RGBA', (1000, 1000), (255,255,255,255))
    draw = ImageDraw.Draw(img)
    for node in environment.nodes:
        elders = [n.id for n in environment.initial_nodes]
        color = '#0000FF' if node.id in elders else '#000000'
        color = color if node.reliability_prob == 1.0 else '#FF0000'
        draw.rectangle(xy=((node.coordinate.x-25, config.map_height-node.coordinate.y-25), (node.coordinate.x + 25, config.map_height-node.coordinate.y + 25)),
                       outline=color, fill=(255,255,255), width=5)
        draw.text(xy=(node.coordinate.x, config.map_height-node.coordinate.y),text=f'{len(node.instances)}',fill=(0,0,0),align='center',stroke_width=1,font_size=14,anchor='mb')
        draw.text(xy=(30, 30),text=f'{id}',fill=(0,0,0),align='center',stroke_width=1,font_size=20,anchor='mb')
        if node.id in elders:
            draw.circle(xy=((node.coordinate.x, config.map_height-node.coordinate.y)),
                           outline=color, radius=node.coverage)

    sampler = qmc.Halton(d=2, scramble=False)
    sample = sampler.random(n=config.initial_instance)
    for point in sample:
        draw.circle(xy=((point[0]*1000,point[1]*1000)),
                    fill=(0,0,0), radius=1)
    for user in environment.users:
        draw.circle(xy=(user.coordinate.x, config.map_height-user.coordinate.y),fill="#000000",radius=10)

    if not os.path.exists(f'./graphics'):
        os.makedirs(f'./graphics')
    if not os.path.exists(f'./graphics/{environment.filePath}'):
        os.makedirs(f'./graphics/{environment.filePath}')
    if not os.path.exists(f'./graphics/{environment.filePath}/{environment.seed}'):
        os.makedirs(f'./graphics/{environment.filePath}/{environment.seed}')
    if not os.path.exists(f'./graphics/{environment.filePath}/{environment.seed}/{episode}'):
        os.makedirs(f'./graphics/{environment.filePath}/{environment.seed}/{episode}')
    img.save(f'./graphics/{environment.filePath}/{environment.seed}/{episode}/{id}.png')

def graph(algorithm_names, environments, episode):
    # plt.show()
    fig, (ax1, ax2) = plt.subplots(2, 1)

    colors = ['#5975A4', '#CC8963', '#5F9E6E', '#B55D60', '#857AAB', '#857AAB']

    for i in range(len(environments)):
        path = f"./result/{environments[i].filePath}/eval.csv"
        # path = f"./result/2024-07-22-14-24-15-1/eval.csv"

        result = pd.read_csv(path)
        ax1.plot(result['success']*100/(result['total_received']), label=algorithm_names[i])
        ax2.plot(result['num_instance'], label=algorithm_names[i])
    ax1.legend()
    ax2.legend()
    plt.savefig(f"./result/{environments[0].filePath}/{episode}.png")
    #plt.show()
    plt.close()

