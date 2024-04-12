from setuptools import setup, find_packages
setup(
name='software_development_project_team1',
version='0.1.6',
author='Laetitia Guérout, Théo Berthet, Lucas Bodelle & Synne Trettenes',
author_email='laetitia.guerout@insa-lyon.fr, theo.berthet@insa-lyon.fr, lucas.bodelle@insa-lyon.fr, synne-moe.trettenes@insa-lyon.fr',
description='Software to help victims portray a culprit using an autoencoder model trained on the CelebA dataset and a genetic algorithm',
packages=find_packages(),
classifiers=[
'Programming Language :: Python :: 3',
'License :: OSI Approved :: MIT License',
'Operating System :: OS Independent',
],
python_requires='>=3.6',
include_package_data=True,
install_requires = [
    'numpy==1.24.0',
'matplotlib==3.7.5',
'pandas==1.4.4',
'scipy==1.10.0',
'scikit-image==0.21.0',
'keras==2.13.1',
'tensorflow==2.13.1'
]
)