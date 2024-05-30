# Usage

After setting the working directory to the root project folder, install all the requirements using:
```
pip install .
```

To run the code, it requires a config file that specifies the experiment configurations (see [here](data/example.yaml)), 
an identifier that determines folder/file names, and a port for tensorboard. These can be specified as flags from the terminal as follows:

```
python .\src\main.py --config .\data\transfer_experiment.yaml --run-id bankheist_pretrained_on_pong --port 6006
```

A copy of the config file (for future reference), tensorboard logs, plots and return data will be saved in the 
"data\<run-id>\" directory. The neural network models are by default saved in this directory as well, but can be saved 
elsewhere if specified otherwise in the config file, as is the case with transfer_experiment.yaml.

Previously executed return curves can be loaded to skip running the same experiments repeatedly. The return curves for 
each experiment are stored as files without extensions at "data\<run-id>\<experiment tag>". Specifying this path as a 
`load_path` under the experiment parameters will load the return curve instead of rerunning the experiment.

## Transfer learning experiments
The above command will execute a transfer learning experiment that consists of pre-training a DDQN agent in the pong 
environment, and transfering the network to a bankheist agent in addition to training a regular DDQN agent in the 
bankheist environment. To change the source and/or target task, different environments can be specified in the config 
file.  Simply changing the environment names will work, but it is recommended to change experiment tags as well for the 
sake of organization, since they are used for file names.

