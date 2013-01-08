#!/usr/bin/env python
"""cgptoolbox setup script."""

# TODO: We might want to add dependencies: 
# http://packages.python.org/distribute/setuptools.html#declaring-dependencies

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(name='cgptoolbox',
        version='0.1',
        description='A toolbox for causally cohesive '
                    'genotype-phenotype modeling',
        maintainer='Jon Olav Vik',
        maintainer_email='jonovik@gmail.com',
        url='http://arken.umb.no/~jonvi/cgptoolbox/',
        download_url='https://github.com/jonovik/cgptoolbox',
        packages = find_packages(),
        )
