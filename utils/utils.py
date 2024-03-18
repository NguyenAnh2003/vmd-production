import argparse
import yaml

def load_config(path: str):
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default=path)
    args = parser.parse_args()
    config_path = args.conf
    param = yaml.safe_load(open(config_path, 'r', encoding="utf8"))
    return param




if __name__ == "__main__":
    pass