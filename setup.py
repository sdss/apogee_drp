#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='apogee_drp',
      version='0.1.0',
      description='SDSS APOGEE data reduction software',
      author='David Nidever, Jon Holtzman, Drew Chojnowski',
      author_email='dnidever@montana.edu',
      url='https://github.com/sdss/apogee_drp',
      packages=find_packages(exclude=["tests"]),
      requires=['numpy','astropy','dlnpyutils','doppler'],
      include_package_data=True,
)
