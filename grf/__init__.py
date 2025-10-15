from typing import Any

from . import common
from . import va2vars
from . import parser
from . import colour

from .grf import *
from .constants import *
from .vox import VoxReader, VoxFile, VoxTrainFile
from .sprites import ResourceAction, Sprite, ImageSprite, ImageFile, FileSprite, Sound, RAWSound, \
    PaletteRemap, EMPTY_SPRITE, fix_palette, quantize, Uncacheable, ResourceFile, \
    PythonFile, WithMask, MaskMode, SpriteWrapper, MoveSprite, QuantizeSprite
from .common import *
from .actions import RVFlags, TrainFlags, CargoClass, train_hpi, train_ton, \
    nml_te, nml_drag, py_property, SpriteRef, SpriteLayout, SpriteLayoutList, \
    GroundSprite, ParentSprite, ChildSprite, TrainFlags, ShipFlags, AIFlags, \
    TrainRunningCost, TrainVisualEffect, IndustryLayout, IndustryTileRef, \
    OldIndustryTileRef, TownGrowthEffect, Eval
from .strings import StringRef, TTDString
from .lib import *
from . import decompile
from .utils import main
from .colour import PALETTE, PIL_PALETTE, SAFE_COLOURS, ALL_COLOURS, WATER_COLOURS, DEFAULT_BRIGHTNESS, \
    CC_COLOURS, WIN_TO_DOS, srgb_to_linear, linear_to_srgb, srgb_to_oklab, oklab_to_srgb, \
    oklab_find_best_colour, oklab_blend, srgb_find_best_colour, make_palette_image, \
    srgb_color_distance, openttd_adjust_brightness

from . import dev


try:
    from setuptools_git_versioning import get_version
    __version__ = str(get_version())
except:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("grf")
    except PackageNotFoundError:
        # package is not installed
        pass


def __getattr__(name: str) -> Any:
    if name in colour.LAZY_CONSTANT_GENERATORS:
        return getattr(colour, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")