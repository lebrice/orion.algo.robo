#!/usr/bin/env python
"""Installation script for `orion.algo.robo`."""
import os

from setuptools import setup

import versioneer

repo_root = os.path.dirname(os.path.abspath(__file__))

tests_require = ["pytest>=3.0.0", "pytest-mock"]

extras_require = {
    "test": tests_require,
}

setup_args = dict(
    name="orion.algo.robo",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="RoBO Wrapper for Oríon",
    long_description=open(os.path.join(repo_root, "README.rst")).read(),
    license="BSD-3-Clause",
    author="Epistímio",
    author_email="xavier.bouthillier@umontreal.ca",
    url="https://github.com/Epistimio/orion.algo.robo",
    packages=["orion.algo.robo"],
    package_dir={"": "src"},
    python_requires=">=3.7",
    include_package_data=True,
    entry_points={
        "BaseAlgorithm": [
            "robo_gp = orion.algo.robo.gp:RoBO_GP",
            "robo_gp_mcmc = orion.algo.robo.gp:RoBO_GP_MCMC",
            # TODO: Temporarily disabled.
            # "robo_randomforest = orion.algo.robo.randomforest:RoBO_RandomForest",
            "robo_dngo = orion.algo.robo.dngo:RoBO_DNGO",
            "robo_bohamiann = orion.algo.robo.bohamiann:RoBO_BOHAMIANN",
            "robo_ablr = orion.algo.robo.ablr.ablr:RoBO_ABLR",
        ],
    },
    install_requires=[
        "orion>=0.2.2",
        "numpy",
        "torch>=1.2.0",
        "pyyaml<6.0.0",  # NOTE: This is because of George uses `yaml.load` without passing a Loader
        "pybind11",
        "Jinja2",
        "tqdm",
        "robo @ git+https://github.com/automl/RoBO.git@91366b12a1a3deb8e80dd08599e0eaf4df28adc1",
        # DNGO depends on older versions (< 3.0.0), otherwise we have to rewrite the train method:
        "emcee<3.0.0",
    ],
    tests_require=tests_require,
    setup_requires=["setuptools", "pytest-runner>=2.0,<3dev"],
    extras_require=extras_require,
    # "Zipped eggs don't play nicely with namespace packaging"
    # from https://github.com/pypa/sample-namespace-packages
    zip_safe=False,
)

setup_args["keywords"] = [
    "Machine Learning",
    "Deep Learning",
    "Distributed",
    "Optimization",
]

setup_args["platforms"] = ["Linux"]

setup_args["classifiers"] = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
] + [("Programming Language :: Python :: %s" % x) for x in "3 3.6 3.7 3.8 3.9".split()]

if __name__ == "__main__":
    setup(**setup_args)
