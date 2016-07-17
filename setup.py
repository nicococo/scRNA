try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'scRNA - scRNA-seq clustering toolbox',
    'url': 'https://github.com/nicococo/scRNA',
    'author': 'Nico Goernitz & Bettina Mieth',
    'author_email': 'nico.goernitz@tu-berlin.de',
    'version': '0.1',
    'install_requires': ['nose', 'cvxopt','scikit-learn','numpy', 'scipy'],
    'packages': ['scRNA'],
    'scripts': ['bin/scRNA-nmf.sh','bin/scRNA-generate-data.sh','bin/scRNA-evaluate.sh','bin/scRNA-sc3.sh','bin/scRNA-mtl-sc3.sh'],
    'name': 'scRNA',
    'classifiers':['Intended Audience :: Science/Research',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7']
}

setup(**config)