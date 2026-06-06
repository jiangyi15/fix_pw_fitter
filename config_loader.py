import yaml

def load_config(filename):
    if isinstance(filename, dict):
        return filename
    with open(filename) as f:
        ret = yaml.full_load(f)
    return ret

class ConfigLoader:
    def __init__(self, config):
        self.config = load_config(config)

