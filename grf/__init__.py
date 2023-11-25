from . import common
from . import va2vars
from . import parser

from .grf import *
from .constants import *
from .vox import VoxReader, VoxFile, VoxTrainFile
from .sprites import ResourceAction, Sprite, ImageSprite, ImageFile, FileSprite, Mask, FileMask, ImageMask, Sound, RAWSound, PaletteRemap, EMPTY_SPRITE, open_image, find_best_color, fix_palette, combine_fingerprint, ResourceFile, PythonFile
from .common import *
from .actions import RVFlags, TrainFlags, CargoClass, train_hpi, train_ton, \
    nml_te, nml_drag, py_property, SpriteRef, SpriteLayout, SpriteLayoutList, \
    GroundSprite, ParentSprite, ChildSprite, TrainFlags, ShipFlags, AIFlags, \
    TrainRunningCost, TrainVisualEffect, IndustryLayout, IndustryTileRef, \
    OldIndustryTileRef, TownGrowthEffect
from .strings import StringRef, TTDString
from .lib import *
from . import decompile
from .utils import main


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
