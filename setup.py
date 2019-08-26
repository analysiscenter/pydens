"""
PyDEns is framework for solving Partial and Ordinary Differential Equations using neural
networks.
"""
import re
from setuptools import setup, find_packages


with open('pydens/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='pydens',
    packages=find_packages(exclude=('tests', )),
    version=version,
    url='https://github.com/analysiscenter/pydens.git',
    license='Apache License 2.0',
    author='Data Analysis Center',
    author_email='akoriagin@nes.ru',
    description='Framework for solving differential equations with deep learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    platforms='any',
    install_requires=[
        'numpy>=1.10',
        'dill>=0.2.7',
        'tqdm>=4.19.7',
        'scipy>=0.19.1',
        'scikit-image>=0.13.1',
        'numba>=0.42',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering'
    ],
)
