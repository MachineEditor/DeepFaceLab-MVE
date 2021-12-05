from .ModelBase import ModelBase
from pathlib import Path

def import_model(model_class_name):
    module = __import__('Model_'+model_class_name, globals(), locals(), [], 1)
    return getattr(module, 'Model')


def get_config_schema_path():
    config_path = Path(__file__).parent.absolute() / Path("config_schema.json")
    return config_path