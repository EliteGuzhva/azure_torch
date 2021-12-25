# imports
import yaml
import argparse
from pathlib import Path
from azureml.core import Workspace, ScriptRunConfig, Experiment, Environment, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import PyTorchConfiguration
from azureml.core.authentication import AzureCliAuthentication

# args
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="cifar_single_test", type=str,
                    help="Run configuration specified in config/{name}.yaml")
args = parser.parse_args()

# read config
config: dict
with open(f"config/{args.config}.yaml", "r") as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)
        exit()

# get workspace
cli_auth = AzureCliAuthentication()
ws = Workspace.from_config(auth=cli_auth)

# get root of git repo
prefix = Path(__file__).parent

# training script
source_dir = str(prefix.joinpath("src"))
script_name = config['train_script']

# compute target
cluster_name = config['cluster_name']
try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size=config['vm_size'], 
                                                           max_nodes=config['num_nodes'])

    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# get environment
env = Environment.get(ws, name=config['environment_name'])

# create azureml dataset
data_root = config['data_dir']
datastore = ws.get_default_datastore()
dataset = Dataset.File.upload_directory(
    src_dir=data_root, target=(datastore, data_root)
)

# create distributed config
distr_config = PyTorchConfiguration(
    node_count=config['num_nodes'], process_count=config['num_processes'])

# create args
script_args = [
        "--data-dir", dataset.as_download(),
        "--epochs", config['epochs'],
        "--batch-size", config['batch_size']
    ]

# create job config
src = ScriptRunConfig(
    source_directory=source_dir,
    script=script_name,
    arguments=script_args,
    compute_target=compute_target,
    environment=env,
    distributed_job_config=distr_config,
)

# submit job
run = Experiment(ws, config['experiment_name']).submit(src)
run.wait_for_completion(show_output=True)
