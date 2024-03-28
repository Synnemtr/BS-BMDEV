from setuptools import setup, find_packages
setup(
name='Software development project',
version='0.1.0',
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
)