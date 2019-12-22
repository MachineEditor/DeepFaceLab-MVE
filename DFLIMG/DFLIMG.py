from pathlib import Path

from .DFLJPG import DFLJPG
from .DFLPNG import DFLPNG

class DFLIMG():
    
    @staticmethod
    def load(filepath, loader_func=None):
        if filepath.suffix == '.png':
            return DFLPNG.load( str(filepath), loader_func=loader_func )
        elif filepath.suffix == '.jpg':
            return DFLJPG.load ( str(filepath), loader_func=loader_func )
        else:
            return None
