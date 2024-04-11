Installation tutorial
=====================


Download from PyPI
------------------

In the terminal : ``pip install software-development-project-team1``

To be sure to have the appropriate version of modules, in the project folder, run in the terminal: 
``pip install -r requirements.txt``

You can now use the Python module and the software in the terminal using : 
``python3``
``import software-development-project-team1``
``software-development-project-team1.identify_attacker``


Download from Github
--------------------

In the terminal : ``git clone https://github.com/Synnemtr/BS-BMDEV.git``

Or you can go to this `link`_ and clone the directory from the GitHub website :

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

