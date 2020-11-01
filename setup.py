#!/usr/bin/env python

from setuptools import setup, find_packages

# version fetch
from tiler import __version__

# readme fetch
with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='tiler',
      version=__version__,
      description='Fast N-dimensional numpy array tiling and merging with overlapping, padding and tapering support',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT',
      author='the-lay',
      author_email='ilja.gubin@gmail.com',
      url='https://github.com/the-lay/tiler',
      platforms=['any'],
      install_requires=[
          'numpy',
          'scipy',
          'tqdm'
      ],
      packages=find_packages(),
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3 :: Only',
      ]
)

