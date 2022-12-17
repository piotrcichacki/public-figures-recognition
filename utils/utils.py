import yaml


def load_yaml(file_path):
    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as err:
            print(err)
