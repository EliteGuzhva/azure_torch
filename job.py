# imports
import argparse
from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import PyTorchConfiguration

# args
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
parser.add_argument("--batch-size", default=16, type=int, help="mini batch size")
parser.add_argument("--train-script", default="train.py", type=str, help="train script filename")
parser.add_argument("--data-dir", default="cifar-10", type=str, help="dataset directory")
parser.add_argument("--experiment-name", default="azure-pytorch", type=str, help="experiment name")
parser.add_argument("--cluster-name", default="gpu-cluster", type=str, help="cluter name")
parser.add_argument("--vm-size", default="STANDARD_NC6", type=str, help="azureml compute target configuration")
parser.add_argument("--num-nodes", default=1, type=int, help="number of nodes in a cluster")
parser.add_argument("--environment-name", default="AzureML-PyTorch-1.6-GPU", type=str, help="Environment name")
args = parser.parse_args()

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).parent

# training script
source_dir = str(prefix.joinpath("src"))
script_name = args.train_script

# compute target
cluster_name = args.cluster_name
try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size=args.vm_size, 
                                                           max_nodes=args.num_nodes)

    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# get environment
env = Environment.get(ws, name=args.environment_name)

# create azureml dataset
data_root = args.data_dir
datastore = ws.get_default_datastore()
dataset = Dataset.File.upload_directory(
    src_dir=data_root, target=(datastore, data_root)
)

# create distributed config
distr_config = PyTorchConfiguration(node_count=args.num_nodes)

# create args
script_args = [
        "--data-dir", dataset.as_download(),
        "--epochs", args.epochs,
        "--batch-size", args.batch_size
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
run = Experiment(ws, args.experiment_name).submit(src)
run.wait_for_completion(show_output=True)

