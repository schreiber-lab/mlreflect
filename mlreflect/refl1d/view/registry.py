from .layer import NoInteractor, SlabInteractor
from .monoi import FreeInterfaceInteractor, FreeLayerInteractor
from .polymeri import PolymerBrushInteractor
from ..model import Slab
from ..mono import FreeInterface, FreeLayer
from ..polymer import PolymerBrush


# ======================== LayerInteractor factory ====================
# Associate layers with layer interactors through function
#     interactor(profile,layer)
# New layer interactors can be registered with
#     make_interactor[layer_class] = interactor_class
# =====================================================================
class InteractorFactory:
    """
    Given a layer, find the associated interactor.
    """

    def __init__(self):
        self._registry = {}

    def __setitem__(self, layer_class, interactor_class):
        self._registry[layer_class] = interactor_class

    def __call__(self, profile, layer):
        constructor = self._registry.get(layer.__class__, NoInteractor)
        return constructor(profile, layer)


interactor = InteractorFactory()
interactor[Slab] = SlabInteractor
interactor[PolymerBrush] = PolymerBrushInteractor
interactor[FreeInterface] = FreeInterfaceInteractor
interactor[FreeLayer] = FreeLayerInteractor