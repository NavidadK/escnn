

from .gspace import GSpace
from .r0 import *
from .r2 import *
from .r2_rothue import *
from .hue import *
from .r3 import *

# TODO move this sub-package to PyTorch part (e.g. under nn)

__all__ = [
    "GSpace",
    "GSpace0D",
    "Hue2D",
    "GSpace2D",
    "RotHue2D",
    "GSpace3D",
    # R0
    "no_base_space",
    # Hue
    "hueOnR2",
    # R2
    "rot2dOnR2",
    "rotHueOnR2",
    "flipRot2dOnR2",
    "flip2dOnR2",
    "trivialOnR2",
    # R3
    "flipRot3dOnR3",
    "rot3dOnR3",
    "fullIcoOnR3",
    "icoOnR3",
    "fullOctaOnR3",
    "octaOnR3",
    "dihedralOnR3",
    "rot2dOnR3",
    "conicalOnR3",
    "fullCylindricalOnR3",
    "cylindricalOnR3",
    "mirOnR3",
    "invOnR3",
    "trivialOnR3"
]
