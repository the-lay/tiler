import sys
from tiler.tiler import Tiler
from tiler.merger import Merger

__version__ = '0.2.0'
__all__ = ['Tiler', 'Merger']

# Import README file as a module general docstring, only when generating documentation
# We also modify it to make it prettier
if "pdoc" in sys.modules:  # pragma: no cover
    with open('README.md', 'r') as f:
        _readme = f.read()
        _readme = _readme.split("# tiler", 1)[1]  # remove header
        _readme = _readme.replace('misc/teaser/tiler_teaser.png', 'tiler_teaser.png')  # replace image path
        __doc__ = _readme
