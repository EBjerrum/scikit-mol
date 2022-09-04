from setuptools import setup

setup(name='scikit_mol',
      version='0.1',
      description='scikit-learn classed for molecule transformation',
      #url='',
      author='Esben Jannik Bjerrum',
      author_email='esben@cheminformania.com',
      license='Apache 2.0',
      packages=['scikit_mol'],
      install_requires=[
          'rdkit',
          'numpy',
          'scikit-learn'
      ],
      zip_safe=False,
      )
