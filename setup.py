from setuptools import setup, find_packages

requirements = ['numpy>=1.14.2', 'pandas>=0.22.0', 'anndata>=0.6.10', 'matplotlib>=3.0.0', 'scipy>=1.0.0', 'scanpy>=1.3.1', 'scikit-learn>=0.20.0', 'hub-toolbox>=2.3.1', 'cvxpy>=1.0.6']

setup(name='popcom',
	install_requires=requirements,
	author="Yijie Wang",
	author_email='yijie.wang@nih.gov',
	version='0.1.0',
	description='PopCom is a new method for the identification of  sub-populations of cells present within individual single cell experiments and mapping of these sub-populations across the experiments.',
	url='https://github.com/wyjhxq/popcom',
	license='GNU General Public License v3',
	packages=['popcom'],
	zip_safe=False)
