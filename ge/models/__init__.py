from .deepwalk import DeepWalk
from .node2vec import Node2Vec
from .struc2vec import Struc2Vec

__all__ = ["DeepWalk", "Node2Vec", "Struc2Vec"]

try:
    from .line import LINE

    __all__.append("LINE")
except ImportError:
    LINE = None

try:
    from .sdne import SDNE

    __all__.append("SDNE")
except ImportError:
    SDNE = None
