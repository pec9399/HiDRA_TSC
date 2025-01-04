import logging.config
import os, sys, gc
from datetime import datetime

import config
from experiment import Experiment

os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == '__main__':
    argv = sys.argv

    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    logging.config.fileConfig(os.getcwd() + '/logging.ini')

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    start_time = datetime.now()
    if len(argv) == 1:
        print(f'Running Experiment\n{now} {config.random_drop}')
        experiment = Experiment()

        for i in range(1,config.iterations+1):
            now2 = datetime.now()
            config.random.seed(i)
            experiment.run(i, now)
            gc.collect()
            end_time2 = datetime.now()
            duration2 = end_time2 - now2
            print(f'Finished experiment in {duration2.total_seconds()}')
    elif len(argv) > 3:
        if argv[1] == 'evaluate':
            print(f'Running evaluation\n{now}')

            model_path = argv[2]
            # model_path = '2024-10-20-10-44-51-1'
            # model_path = '2024-10-09-19-36-55-1'
            print(f'Model: {model_path}')

            seed = int(argv[3])
            config.initial_instance = int(argv[4])
            config.random_drop = int(argv[5])
            config.device = argv[6]
            train_num = argv[7]
            experiment = Experiment()
            
            config.random.seed(seed)
            experiment.run(seed, now, model_path, train_num)
            gc.collect()
        else:
            print(f'Running Experiment\n{now}')
            experiment = Experiment()
            seed = int(argv[1])
            config.initial_instance = int(argv[2])
            config.random_drop = int(argv[3])
            config.device = argv[4]
            config.random.seed(seed)
            experiment.run(seed, now)
            gc.collect()
    end_time = datetime.now()
    duration = end_time - start_time
    print(f'Finished experiment in {duration.total_seconds()}')
