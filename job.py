# imports
import os
import urllib
import tarfile
from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import PyTorchConfiguration

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).parent

# training script
source_dir = str(prefix.joinpath("src"))
script_name = "train.py"

# azure ml settings
environment_name = "AzureML-PyTorch-1.6-GPU"  # using curated environment
experiment_name = "pytorch-cifar10-distributed-example"

# compute target
cluster_name = "gpu-cluster"
try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                           max_nodes=2)

    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# get environment
env = Environment.get(ws, name=environment_name)

# download and extract cifar-10 data
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename = "cifar-10-python.tar.gz"
data_root = "cifar-10"
filepath = os.path.join(data_root, filename)

if not os.path.isdir(data_root):
    os.makedirs(data_root, exist_ok=True)
    urllib.request.urlretrieve(url, filepath)
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=data_root)
    os.remove(filepath)  # delete tar.gz file after extraction

# create azureml dataset
datastore = ws.get_default_datastore()
dataset = Dataset.File.upload_directory(
    src_dir=data_root, target=(datastore, data_root)
)

# create distributed config
distr_config = PyTorchConfiguration(node_count=1)

# create args
args = ["--data-dir", dataset.as_download(), "--epochs", 25]

# create job config
src = ScriptRunConfig(
    source_directory=source_dir,
    script=script_name,
    arguments=args,
    compute_target=compute_target,
    environment=env,
    distributed_job_config=distr_config,
)

# submit job
run = Experiment(ws, experiment_name).submit(src)
run.wait_for_completion(show_output=True)

