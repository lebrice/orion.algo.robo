===============
orion.algo.robo
===============


.. |pypi| image:: https://img.shields.io/pypi/v/orion.algo.robo
    :target: https://pypi.python.org/pypi/orion.algo.robo
    :alt: Current PyPi Version

.. |py_versions| image:: https://img.shields.io/pypi/pyversions/orion.algo.robo.svg
    :target: https://pypi.python.org/pypi/orion.algo.robo
    :alt: Supported Python Versions

.. |license| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: BSD 3-clause license

.. |rtfd| image:: https://readthedocs.org/projects/orion.algo.robo/badge/?version=latest
    :target: https://orion.algo-robo.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |codecov| image:: https://codecov.io/gh/Lucasc-99/orion.algo.robo/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/Lucasc-99/orion.algo.robo
    :alt: Codecov Report

.. |travis| image:: https://travis-ci.org/Lucasc-99/orion.algo.robo.svg?branch=master
    :target: https://travis-ci.org/Lucasc-99/orion.algo.robo
    :alt: Travis tests


----

This ``orion.algo`` plugin was generated with `Cookiecutter`_ along with `@Epistimio`_'s `cookiecutter-orion.algo`_ template.

See Orion : https://github.com/Epistimio/orion


Installation
------------

The RoBO wrapper is currently only supported on Linux.

Before installing RoBO, make sure you have libeigen and swig installed. On ubuntu, you can install
them with `apt-get`::

    $ sudo apt-get install libeigen3-dev swig

One of the dependencies of RoBO does not declare its dependencies and therefore we need
to install these dependencies first. The order of dependencies in `requirements.txt` reflects this
order. To install them sequentially, use the following command::

    $ curl -s https://git.io/JLnCA | grep -v "^#" | xargs -n 1 -L 1 pip install

Finally, you can install this package using PyPI::

    $ pip install orion.algo.robo


Usage
-----

Contribute or Ask
-----------------

Do you have a question or issues? Do you want to report a bug or suggest a feature? Name it! Please
contact us by opening an issue in our repository below and checkout Oríon's
`contribution guidelines <https://github.com/Epistimio/orion/blob/develop/CONTRIBUTING.md>`_:

- Issue Tracker: `<https://github.com/Epistimio/orion.algo.robo/issues>`_
- Source Code: `<https://github.com/Epistimio/orion.algo.robo>`_

Start by starring and forking our Github repo!

Thanks for the support!

Citation
--------

If you use this wrapper for your publications, please cite both
`RoBO <https://github.com/automl/RoBO#citing-robo>` and 
`Oríon <https://github.com/epistimio/orion#citation>`.


License
-------

Distributed under the terms of the BSD-3-Clause license,
"orion.algo.robo" is free and open source software.


.. _`Cookiecutter`: https://github.com/audreyr/cookiecutter
.. _`@Epistimio`: https://github.com/Epistimio
.. _`cookiecutter-orion.algo`: https://github.com/Epistimio/cookiecutter-orion.algo
.. _`orion`: https://github.com/Epistimio/orion
