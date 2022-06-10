======================
RoBO Wrapper for Oríon
======================


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

.. |codecov| image:: https://codecov.io/gh/Epistimio/orion.algo.robo/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/Epistimio/orion.algo.robo
    :alt: Codecov Report

.. |github-actions| image:: https://github.com/Epistimio/orion.algo.robo/workflows/build/badge.svg?branch=master&event=pull_request
    :target: https://github.com/Epistimio/orion.algo.robo/actions?query=workflow:build+branch:master+event:schedule
    :alt: Github actions tests

----

This wrapper provides access through `Oríon`_ to several Bayesian optimization algorithms
in the library `RoBO`_.

This ``orion.algo`` plugin was generated with `Cookiecutter`_ along with `@Epistimio`_'s
`cookiecutter-orion.algo`_ template.

Installation
------------

The RoBO wrapper is currently only supported on Linux.

Before installing RoBO, make sure you have ``libeigen`` and ``swig`` installed.
On ubuntu, you can install them with ``apt-get``, like so:

.. code-block:: console

    $ sudo apt-get install libeigen3-dev swig

The RoBO wrapper can then be installed, either from source, like so:

.. code-block:: console

    $ pip install git+https://github.com/Epistimio/orion.algo.robo

Or from PyPI:

.. code-block:: console

    $ pip install orion.algo.robo


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
`RoBO <https://github.com/automl/RoBO#citing-robo>`__ and
`Oríon <https://github.com/epistimio/orion#citation>`__. Please also cite
the papers of the algorithms you used, such as DNGO or BOHAMIANN. See
the documentation of the algorithms to find corresponding original papers.

License
-------

Distributed under the terms of the BSD-3-Clause license,
``orion.algo.robo`` is free and open source software.


.. _`Cookiecutter`: https://github.com/audreyr/cookiecutter
.. _`@Epistimio`: https://github.com/Epistimio
.. _`cookiecutter-orion.algo`: https://github.com/Epistimio/cookiecutter-orion.algo
.. _`Oríon`: https://github.com/Epistimio/orion
.. _`RoBO`: https://github.com/automl/robo
