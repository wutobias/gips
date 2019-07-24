from setuptools import setup, Extension, find_packages
import os
import numpy as np

__version__ = "0.6"

# define the extension module
extensions = []
extensions.append(Extension('gips.utils._read_ext',
                            sources=['./src/_read_ext.c'],
                            include_dirs=[np.get_include()],
                            language='c'))

extensions.append(Extension('gips.grid_solvent._spatial_ext',
                            sources=['./src/_spatial_ext.c',
                                     './src/Vec.c'],
                            include_dirs=[np.get_include()],
                            language='c'))

extensions.append(Extension('gips.utils._read_ext',
                            sources=['./src/_read_ext.c'],
                            include_dirs=[np.get_include()],
                            language='c'))

extensions.append(Extension('gips.gistmodel._numerical_ext',
                            sources=['./src/_numerical_ext.c',
                                     './src/gist_functionals.c',
                                     './src/gist_functionals_check.c'],
                            include_dirs=[np.get_include()],
                            extra_compile_args=['-fopenmp'],
                            extra_link_args=['-lgomp'],
                            language='c'))

setup(name='gips',
    author='Tobias Wulsdorf',
    author_email='tobias.wulsdorf@gmail.com',
    description='gips: Modelling Toolkit for Solvation-based Structure Affinity Relationships. \
Developed by Tobias Wulsdorf in the group of Prof. Gerhard Klebe at Marburg University.',
    version=__version__,
    license='MIT',
    platforms=['Linux'],
    packages=find_packages(),
    ext_modules=extensions,
    zip_safe=False,
    entry_points={
        'console_scripts':
        ['run_gips  = gips.scripts.run_gips:entry_point']}, )
