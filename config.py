import argparse
import yaml 

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default='./configs/fungi.yaml',
                        help='configurations for training')
    return parser.parse_args()
args = parse_config()
with open(f'{args.config}', "r") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)