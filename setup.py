#!/usr/bin/env python

try:
	from setuptools import setup
except BaseException:
	from distutils.core import setup

# from setuptools import setup, find_packages
from attnptrns import __version__

setup(name='attnptrns',
      version=__version__,
      description='Attention patterns',
      author='Théo Gigant',
      author_email='theo.gigant@l2s.centralesupelec.fr',
      url='https://github.com/giganttheo/attention_patterns',
      packages=['attnptrns'],
     )