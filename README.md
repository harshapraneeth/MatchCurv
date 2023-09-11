# MatchCurv

## File Structure

- **code:**

    - **_db_cert:** This directory contains the necessary database certificate for validating Firebase Cloud Database credentials. However, it is not required for the code to function if the `ignore_database` setting in the config file is set to True. Please note that these credentials are specific to my Firebase account and are not provided with the code. Attempting to use the `Database.py` class without the correct credentials will result in failure. To ensure successful experiments, set `ignore_database` to True in the experiment's configuration.

    - **_graph_presets:** This directory holds graph presets used for experiments to simulate various random topologies. These presets are essential for both simulations and running experiments on Raspberry Pi devices. Different graph presets represent different topologies; be sure to specify the one used in the config file.

    - **_model_presets:** Similar to graph presets, this directory contains initial model weights. While specifying a model preset is not mandatory, you can do so in the config file if required.

    - **_random_presets:** This directory contains random presets, which replace the default pseudo-random generator. These presets are necessary for the successful execution of experiments.

    - **configs:** The `configs` folder and its subfolders store configuration files for experiments. To run an experiment, create a subfolder in this directory and generate one or more JSON files following the existing examples. Each config file in a subfolder represents an experiment with its specific settings. Additionally, there is an option to define a search space. For instance, if you need to find an optimal learning rate, create a JSON file named `_search_space.json` and specify the parameters and their corresponding values to be searched.

    - **datasets:** This directory contains samples from the MNIST and Fashion-MNIST datasets. Each dataset's samples are organized into individual files and are extracted during the "preheat" step. The specific dataset used in an experiment is specified in the config file.

    - **logs:** Experiment logs are stored in this folder, with each experiment having its own subfolder as specified in the config file.

    - **results:** The results of each experiment are saved in this folder, with each experiment having its designated subfolder based on the config file.

    - **src:**
        - **dashboard:** This subdirectory contains the code for the project's dashboard. It requires the database certificate for operation. To track the training progress start a live server at the location  of the html file and open it.
        
        - **matchcurv:** Inside this subdirectory, you'll find the code for the framework. Detailed descriptions of each script can be found within the individual files.

    - **presentation:** This directory contains the final presentation (ppt) for the project.

    - **documentation.zip:** This zip file contains the documentation/final paper for the project.

## Running the simulation

**[1]** Create a folder for your experiment configs:

```shell
> cd code/configs
> mkdir experiment1
> cd experiment1
```

**[2]** Create one or more config files as needed, and populate them using existing configs as a reference. You can also create a `_search_space.json` file if required:

```shell
> nano config1.json
> nano config2.json
> nano _search_space.json
```

Here's an example of a config file:

```json
{
    "is_simulation": "True",
    "ip": "",
    "tcp_port": 4000,
    "udp_port": 4001,
    :
    "clear_logs": "True",
    "ignore_logger": "False",
    "ignore_database": "True",
    "sync_delay": 10
}
```
Set `is_simulation` to `True` if the code is running inside a simulation, otherwise `False`.
Fill the `ip` with the device ip if running on PIs, otherwise leave empty.

**[3]** Example of `_search_space.json`: In this example, we are searching for `curv_term` and `random_preset`. Each experiment will replace the `curv_term` and `random_preset` in the config file with values specified in the search space, resulting in multiple experiments:

```json
{
    "curv_term" : [
        [0.01, 0.1, 1], 
        ["0_d_01", "0_d_1", "1"]
    ],
    
    "random_preset" : [
        [
            "../../_random_presets/random_preset_1.b",
            "../../_random_presets/random_preset_2.b"
        ],
        ["preset1", "preset2"]
    ]
}
```
So the total number of experiments would be 3 curv_terms times 2 random_presets times 2 config files, equals to 12. The subsequent list after the search space values is the aliases for the values used in creating the results file.

**[4]** Go back to the directory containing the `Simulation.py` script and edit it to specify the path to your config file(s):

```shell
> cd ../../src/matchcurv
> nano Simulation.py
```

Specify the path to your config file(s) in the `locations` list:

```python
if __name__ == "__main__":
    '''
    This script is run for starting the simulation.
    '''
    CONFIG_PATH = "../../configs/"
    RESULTS_PATH = "../../results/"
    LOGS_PATH = "../../logs/"

    locations = [
        "experiment1/"
    ]
```

**[5]** When everything is configured, run the experiment:

```shell
> python Simulation.py
```

## Running on PIs

Steps 1-4 same as above.

**[5]** When everything is configured, run the experiment:

```shell
> python Device.py path/to/config -1 120 (a few seconds for the process to start, for e.g., 120)
```

## Configuration File

**Simulation Settings:**

- Set the `is_simulation` field to `True` if the program is running in simulation mode, otherwise set it to `False`.
- If the program is running on Raspberry Pi devices, you must specify the `ip` address of the device.
- Ensure that all devices across the network use the same `tcp_port` and `udp_port` settings.

```json
{
    "is_simulation": true,
    "ip": "",
    "tcp_port": 4000,
    "udp_port": 4001,
```

**Dashboard-Related Settings (Not Needed):**

These settings are related to the dashboard and are not required for normal operation.

```json
    "certificate": "!!!REDACTED!!!",
    "database_url": "!!!REDACTED!!!",
```

**Graph Decomposition and Activation Settings:**

- To perform graph decomposition into matchings, leave the `decomposition` field as `"matcha"`, which is the default value and should not be changed.
- To specify the method for computing activation probabilities, set the `activations` field to either `"random"` or `"matcha"`.
- You can control the communication budget using the `comm_budget` field, which should be set to a value between 0 and 1.

```json
    "decomposition": "matcha",
    "activations": "random",
    "comm_budget": 0.25,
```

**Introducing System Heterogeneity:**

- To introduce system heterogeneity, specify the percentage of straggler devices using the `stragglers` field.
- You can also specify the lag time for straggler devices using the `lag` field.
- Set the `num_seconds` field to a value greater than the lag time. In experiments, it's typically set to twice the lag time, assuming stragglers perform half the number of epochs.

```json
    "stragglers": 0.5,
    "lag": 10,
    "num_epochs": -1,
    "num_seconds": 20,
```

If estimating the lag and num_seconds is tricky, an alternative approach is provided. In this case, set `lag` to `1`, `num_seconds` to `1` and `num_epochs` to the desired value. Then uncomment the following code section in `Device.py`:

```python
'''
Train the model.
'''

self.model.train(
    num_seconds = -1,
    num_epochs = num_epochs // (2 if self.straggler else 1),
    batch_size = self.config["batch_size"]
)
```

**Model Parameters:**

- Specify the model type as either `"MLP"` or `"LeNet5"` in the `model` field.
- Use the `model_args` field to specify the number of units and layers.
- Configure other model-related settings such as `num_rounds`, `num_epochs`. 

The `l2_term` is the L2 regularization term, `prox_term` is the constant for FedProx and is deprecated. Finally, `curv_term` is set to a non zero value to use FedCurv penalty.

```json
    "model": "MLP",
    "model_args": [128, 128],
    "num_rounds": 20,
    "num_epochs": 50,
    "num_seconds": -1,
    "batch_size": 128,
    "input_shape": [28, 28, 1],
    "num_outputs": 10,
    "learning_rate": 0.01,
    "l2_term": 0,
    "prox_term": 0,
    "curv_term": 0,
```

**Statistical Heterogeneity Configuration:**

To introduce statistical heterogeneity by distributing the dataset among devices:

- Specify the paths to training and testing files.
- Set `label_distribution` to indicate how many labels/classes each device includes samples from.
- Adjust `sample_distribution` to specify the percentage of samples from assigned labels distributed among devices.

```json
    "train_files": "../../datasets/mnist/train/",
    "test_files": "../../datasets/mnist/test/",
    "label_distribution": 2,
    "sample_distribution": 0.5,
```

**Experiment Presets:**

Specify the paths to presets used in experiments:

```json
    "random_preset": "../../_random_presets/random_preset_2.b",
    "graph_preset": "../../_graph_presets/sparse_graph.b",
    "model_preset": "../../_model_presets/mlp_128_128_preset_1.b",
```

**Other Parameters:**

- `logs_location`: Specify the location for storing logs.
- `results_location`: Specify the location for storing results.
- `stdout`: Set to `False` to prevent program outputs from flooding the screen during simulation. For Raspberry Pi deployment, you can set it to `True`.
- `clear_logs`: Set to `True` to clear existing logs.
- `ignore_logger`: Set to `False` to enable logging.
- `ignore_database`: Set to `True` to ignore database-related operations.
- `sync_delay`: Set the delay (in seconds) for `Simulation.py` to spawn all subprocesses before starting training. Adjust as needed based on the number of processes.

```json
    "logs_location": "../../logs/final/",
    "results_location": "../../results/final",
    "stdout": "False",
    "clear_logs": "True",
    "ignore_logger": "False",
    "ignore_database": "True",
    "sync_delay": 10
}
```