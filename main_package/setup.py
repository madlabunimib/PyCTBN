from setuptools import setup, find_packages

setup(name='ctbn',
      version='1.0',
      url='https://github.com/philipMartini/CTBN_Project',
      #license='MIT',
      author='Luca Moretti,Filippo Martini',
      author_email='l.moretti@campus.unimib.it,f.martini@campus.unimib.it',
      description='A Continuous  Time Bayesian Network Library',
      packages=find_packages(exclude=['tests', 'data']),
      long_description=open('README.md').read(),
      zip_safe=False)