import sys

from tiler.merger import Merger
from tiler.tiler import Tiler

__all__ = ["Tiler", "Merger"]

# Import README file as a module general docstring, only when generating documentation
# We also modify it to make it prettier
if "pdoc" in sys.modules:  # pragma: no cover
    with open("README.md", "r") as f:
        _readme = f.read()

        # remove baby logo and header
        _readme = _readme.split("\n", 2)[2]

        # replace teaser image path
        _readme = _readme.replace("misc/teaser/tiler_teaser.png", "tiler_teaser.png")
        _readme = _readme.replace("misc/baby_logo.png", "baby_logo.png")
        __doc__ = _readme
