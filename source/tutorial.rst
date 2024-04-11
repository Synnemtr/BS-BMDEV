Installation tutorial
=====================

Download from Github
--------------------

In the terminal : ``git clone https://github.com/Synnemtr/BS-BMDEV.git``

Or you can go to this `link`_ and clone the directory from the GitHub website.

.. _link: https://github.com/Synnemtr/BS-BMDEV

Launch the software using Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To launch the software using Docker, you need to have Docker Desktop installed on your computer.

On MacOS
""""""""

Open Docker Desktop on your computer. In the Dockerfile in this directory, put the name of the file that should be run when using the ``./run.sh`` command.

Make ``run.sh`` executable: ``chmod +x run.sh``

Run ``./run.sh``

On Windows
"""""""""""

Open Docker Desktop on your computer.

Run ``bash run.sh``

Launch the software from the directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure to have the correct versions of the modules by running in the software folder in the terminal : ``pip install -r requirements.txt``

Still in the software folder, you can then launch the software by running :
``python3 identify_attacker.py``



Download from PyPI
------------------

The PyPI module is not working yet due to issues related to the weights of the project.

In the terminal : ``pip install software_development_project_team1`` or to install it in a specific directory ``pip install software_development_project_team1``

You can now use the Python module and the software in the terminal using : 
``python3 software-development-project-team1.identify_attacker``


