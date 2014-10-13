#!/usr/bin/env python
"""cgptoolbox setup script."""

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
        package_data={"cgp.physmod": ["_cellml/*"]},
        install_requires=[
            "numpy", 
            "scipy", 
            "matplotlib>=1.1.1", 
            "lxml", 
            "rpy2", 
            "ipython[parallel]", 
            "Cython", 
            "pyzmq", 
            "networkx", 
            "tables", 
            "joblib", 
            "nose>=1.2", 
            "nose-exclude",
            "bottle",
            "pysundials>=2.3.0-rc4",
        ],
        extras_require={"docs": ["sphinx>=1.1.3", "docutils"]},
        )
