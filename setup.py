from setuptools import setup, find_packages

setup(name='proteinmpnn',
      version='1.0.0',
      description='Custom install of ProteinMPNN for BigHat',
      author='Rosetta Commons',
      url='https://github.com/dauparas/ProteinMPNN',
    #   scripts=["protein_mpnn_run.py"],
      packages=find_packages(),
      install_requires=['torch'])
