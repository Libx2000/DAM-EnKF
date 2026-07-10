import argparse
import os
import subprocess

from src.utils.utils import str2bool


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NuWa-Assimilation Training')

    parser.add_argument('--multi_data_url',
                        help='Dataset URL for remote training',
                        default='[{}]')                        

    parser.add_argument('--train_url',
                        help='Output URL for remote training',
                        default='')
    
    parser.add_argument('--config_file',
                        type=str,
                        help='Hydra config file')

    parser.add_argument('--trainer',
                        type=str,
                        help='Trainer config file',
                        default='gpu')
    
    parser.add_argument('--model',
                        type=str,
                        help='Model config file',
                        default='climax')

    parser.add_argument('--datamodule',
                        type=str,
                        help='DataModule config file',
                        default='h5forecast')

    parser.add_argument('--paths',
                        type=str,
                        help='Paths config file',
                        default='forecast_hpc')

    parser.add_argument('--openi',
                        type=str2bool,
                        help='Use OpenI platform',
                        default=False)
    
    args, unknown = parser.parse_known_args()

    if args.openi:
        from src.utils.openi import c2net_multidataset_to_env as DatasetToEnv

        data_dir = '/cache/data'  
        train_dir = '/cache/output'

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)  
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        local_rank = int(os.getenv('RANK_ID', '0'))
        device_num = int(os.getenv('RANK_SIZE', '1'))
        
        if device_num == 1:
            DatasetToEnv(args.multi_data_url, data_dir)
        else:
            if local_rank == 0:
                DatasetToEnv(args.multi_data_url, data_dir)
                with open("/cache/download_input.txt", 'w') as f:
                    pass

    subprocess.call([
        "python", os.path.join(os.path.dirname(__file__), "src/train.py"),
        f"trainer={args.trainer}.yaml", 
        f"model={args.model}.yaml",
        f"paths={args.paths}.yaml",
        f"datamodule={args.datamodule}.yaml"
    ])