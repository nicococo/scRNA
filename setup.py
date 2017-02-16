try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Single-cell RNA-seq multitask clustering toolbox',
    'url': 'https://github.com/nicococo/scRNA',
    'author': 'Nico Goernitz, Bettina Mieth, Marina Vidovic, Alex Gutteridge',
    'author_email': 'nico.goernitz@tu-berlin.de',
    'version': '2017.02',
    'install_requires': ['nose','scikit-learn','numpy', 'scipy'],
    'packages': ['scRNA'],
    'package_dir' : {'scRNA': 'scRNA'},
    # 'package_data': {'scRNA': ['gene_names.txt']},
    'scripts': ['bin/scRNA-source.sh', 'bin/scRNA-target.sh', 'bin/scRNA-generate-data.sh'],
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