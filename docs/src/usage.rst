.. _robo_gp:

RoBO Gaussian Process
---------------------

.. code-block:: yaml

    experiment:
        algorithms:
            RoBO_GP:
                seed: 0
                n_initial_points: 20
                maximizer: 'random'
                acquisition_func: 'log_ei'
                normalize_input: True
                normalize_output: False


.. autoclass:: orion.algo.robo.gp.RoBO_GP


.. _robo_gp_mcmc:

RoBO Gaussian Process with MCMC
-------------------------------

.. code-block:: yaml

    experiment:
        algorithms:
            RoBO_GP_MCMC:
                seed: 0
                n_initial_points: 20
                maximizer: 'random'
                acquisition_func: 'log_ei'
                normalize_input: True
                normalize_output: False
                chain_length: 2000
                burnin_steps: 2000


.. autoclass:: orion.algo.robo.gp.RoBO_GP_MCMC
   :exclude-members: build_acquisition_func

.. _robo_random_forest:

RoBO Random Forest
------------------

.. code-block:: yaml

    experiment:
        algorithms:
            RoBO_RandomForest:
                seed: 0
                n_initial_points: 20
                maximizer: 'random'
                acquisition_func: 'log_ei'
                num_trees: 30
                do_bootstrapping: True
                n_points_per_tree: 0
                compute_oob_error: False
                return_total_variance: True


.. autoclass:: orion.algo.robo.randomforest.RoBO_RandomForest
   :exclude-members: build_acquisition_func

.. _robo_dngo:

RoBO DNGO
---------

.. code-block:: yaml

    experiment:
        algorithms:
            RoBO_DNGO:
                seed: 0
                n_initial_points: 20
                maximizer: 'random'
                acquisition_func: 'log_ei'
                normalize_input: True
                normalize_output: False
                chain_length: 2000
                burnin_steps: 2000
                batch_size: 10
                num_epochs: 500
                learning_rate: 1e-2
                adapt_epoch: 5000


.. autoclass:: orion.algo.robo.dngo.RoBO_DNGO
   :exclude-members: build_acquisition_func


.. _robo_bohamiann:


RoBO BOHAMIANN
--------------

.. code-block:: yaml

    experiment:
        algorithms:
            RoBO_BOHAMIANN:
                seed: 0
                n_initial_points: 20
                maximizer: 'random'
                acquisition_func: 'log_ei'
                normalize_input: True
                normalize_output: False
                burnin_steps: 2000
                sampling_method: "adaptive_sghmc"
                use_double_precision: True
                num_steps: null
                keep_every: 100
                learning_rate: 1e-2
                batch_size: 20
                epsilon: 1e-10
                mdecay: 0.05
                verbose: False


.. autoclass:: orion.algo.robo.bohamiann.RoBO_BOHAMIANN
   :exclude-members: build_acquisition_func
