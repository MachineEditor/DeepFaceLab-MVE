from .ModelBase import ModelBase
from .ConverterBase import ConverterBase
from .ConverterMasked import ConverterMasked
from .ConverterImage import ConverterImage

def import_model(name):
    module = __import__('Model_'+name, globals(), locals(), [], 1)
    return getattr(module, 'Model')