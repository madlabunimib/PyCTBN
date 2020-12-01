from setuptools import setup, find_packages

setup(name='PyCTBN',
      version='1.0',
      url='https://github.com/philipMartini/CTBN_Project',
      license='MIT',
      author='Filippo Martini',
      author_email='f.martini@campus.unimib.it',
      description='A Continuous Time Bayesian Network Library',
      packages=find_packages(exclude=['tests', 'data']),
      install_requires=[
          'numpy', 'pandas', 'networkx'],
      dependency_links=['https://github.com/numpy/numpy', 'https://github.com/pandas-dev/pandas',
                        'https://github.com/networkx/networkx'],
      long_description=open('README.md').read(),
      zip_safe=False)
