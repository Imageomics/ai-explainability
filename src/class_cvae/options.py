import yaml

class Options:
    def __init__(self, configs):
        for key in configs:
            setattr(self, key, configs[key])

    def __str__(self):
        out = "=======Options=======\n"
        for key in self.__dict__.keys():
            out += f"{key} => {self.__dict__[key]}\n"
        out += "====================="
        return out

def load_config(path):
    with open(path) as f:
        configs = yaml.safe_load(f)
    return Options(configs)

def add_configs(obj, path):
    with open(path) as f:
        configs = yaml.safe_load(f)
    for key in configs:
        setattr(obj, key, configs[key])