import argparse
import os
import yaml
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class MixteraSlurmConfig:
    job_name: str
    log_dir: Path
    partition: Literal["normal", "debug"]
    ntasks_per_node: int
    gpus_per_task: int
    time: str
    environment: str
    account: str
    exclude: Optional[str] = None


@dataclass
class MixteraServerConfig:
    slurm: MixteraSlurmConfig
    server_dir: Path
    mixtera_dir: Path
    port: int

    @classmethod
    def from_yaml(cls, path: str) -> "MixteraServerConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Parse the 'slurm' section
        slurm_data = data.get("slurm", {})
        slurm_config = MixteraSlurmConfig(
            job_name=slurm_data["job_name"],
            log_dir=Path(slurm_data["log_dir"]),
            partition=slurm_data["partition"],
            ntasks_per_node=slurm_data["ntasks_per_node"],
            gpus_per_task=slurm_data["gpus_per_task"],
            time=slurm_data["time"],
            environment=slurm_data["environment"],
            account=slurm_data["account"]
        )

        # Create the server config
        server_config = cls(
            slurm=slurm_config,
            server_dir=Path(data["server_dir"]),
            mixtera_dir=Path(data["mixtera_dir"]),
            port=data["port"],
        )
        return server_config


@dataclass
class TorchtitanConfig:
    torchtitan_src: Path
    torchtitan_logs: Path
    config_file: Path  # Path to the torchtitan TOML config
    environment_vars: dict
    additional_setup: Optional[Path]
    run_ident: str
    job_name: str
    mixtera_dir: Path

    @classmethod
    def from_yaml(cls, path: str) -> "TorchtitanConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Create the torchtitan config
        torchtitan_config = cls(
            torchtitan_src=Path(data["torchtitan_src"]),
            torchtitan_logs=Path(data["torchtitan_logs"]),
            config_file=Path(data["config_file"]),
            environment_vars=data.get("environment_vars", {}),
            additional_setup=Path(data["additional_setup"])
            if data.get("additional_setup")
            else None,
            run_ident=data.get("run_ident", "run"),
            job_name=data["job_name"],
            mixtera_dir=Path(data["mixtera_dir"]),
        )

        return torchtitan_config


@dataclass
class CombinedConfig:
    server_config: MixteraServerConfig
    client_config: TorchtitanConfig
    total_nodes: int
    
    @property
    def time(self) -> str:
        # Use the longer of the two time limits
        server_hours, server_mins = map(int, self.server_config.slurm.time.split(':')[:2])
        client_hours, client_mins = map(int, self.client_config.slurm.time.split(':')[:2])
        
        server_total = server_hours * 60 + server_mins
        client_total = client_hours * 60 + client_mins
        
        if client_total > server_total:
            return self.client_config.slurm.time
        return self.server_config.slurm.time


def get_no_conda_env():
    env = os.environ.copy()

    conda_prefix = env.get("CONDA_PREFIX", "")
    conda_bin = os.path.join(conda_prefix, "bin")

    keys_to_remove = [key for key in env if "CONDA" in key or "PYTHON" in key]
    for key in keys_to_remove:
        if key in env:
            del env[key]

    paths = env["PATH"].split(os.pathsep)
    paths = [p for p in paths if conda_bin not in p and conda_prefix not in p]
    env["PATH"] = os.pathsep.join(paths)

    return env


def build_sbatch_script(config: CombinedConfig) -> str:
    # Prepare log directories
    server_log_dir = config.server_config.slurm.log_dir
    os.makedirs(server_log_dir, exist_ok=True)
    
    client_log_ident = f"{config.client_config.run_ident}_{config.client_config.job_name}"
    client_log_dir = config.client_config.torchtitan_logs / config.client_config.job_name / client_log_ident
    os.makedirs(client_log_dir, exist_ok=True)
    
    # Build the SBATCH header
    sbatch_header = f"""#!/bin/bash
#SBATCH --job-name=mixtera_combined_{config.client_config.run_ident}
#SBATCH --nodes={config.total_nodes}
#SBATCH --ntasks-per-node={config.server_config.slurm.ntasks_per_node}
#SBATCH --time={config.time}
#SBATCH --output={client_log_dir}/combined_%j.out
#SBATCH --error={client_log_dir}/combined_%j.err
#SBATCH --gpus-per-task={config.server_config.slurm.gpus_per_task}
#SBATCH --account={config.server_config.slurm.account}
#SBATCH --partition={config.server_config.slurm.partition}
#SBATCH --environment={config.server_config.slurm.environment}
"""

    if config.server_config.slurm.exclude:
        sbatch_header += f"#SBATCH --exclude={config.server_config.slurm.exclude}\n"

    # Environment variable definitions for client
    env_var_lines = "\n# Set environment variables for client\n"
    for var_name, value in config.client_config.environment_vars.items():
        env_var_lines += f"export {var_name}={value}\n"

    # Additional setup commands for client if any
    additional_setup = ""
    if config.client_config.additional_setup and config.client_config.additional_setup.exists():
        with open(config.client_config.additional_setup, "r") as f:
            additional_setup = f"# Additional client setup\n{f.read()}\n"

    # Get node information and create log file paths
    node_setup = f"""
# Get node information
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )

echo "All nodes: ${{nodes[@]}}"
echo "Node count: ${{#nodes[@]}}"

nodes_array=("${{nodes[@]}}")
total_nodes=${{#nodes_array[@]}}
echo "total_nodes: ${{total_nodes}}"

head_node=${{nodes_array[0]}}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Server node: $head_node, IP: $head_node_ip"

# Get client head node (first client node) for C10D
client_head_node=${{nodes_array[1]}}
client_head_ip=$(srun --nodes=1 --ntasks=1 -w "$client_head_node" hostname --ip-address)
echo "Client head node: $client_head_node, IP: $client_head_ip"

# Create server log file paths with actual job ID
server_log_file="{server_log_dir}/server_{config.server_config.slurm.job_name}_$SLURM_JOB_ID.log"
server_err_file="{server_log_dir}/server_{config.server_config.slurm.job_name}_$SLURM_JOB_ID.err"

# Create client log file paths
client_log_file="{client_log_dir}/output.log"
client_err_file="{client_log_dir}/output.err"
"""

    # Server setup and execution with redirected output
    server_cmd = f"""
# Start server on the first node with log redirection
echo "Starting server, logs will be written to $server_log_file"
srun --nodes=1 --ntasks=1 -w "$head_node" --environment={config.server_config.slurm.environment} bash -c "
set -x
echo 'Server starting on node: $(hostname)'
echo 'Server directory: {config.server_config.server_dir}'
echo 'Server port: {config.server_config.port}'
echo 'Host IP: '$head_node_ip

pushd {config.server_config.mixtera_dir}
echo 'Installing mixtera package...'
pip install -e .
popd

echo 'Running server:'
CMD='numactl --membind=0-3 python -u -m mixtera.network.server.entrypoint --host '$head_node_ip' --port {config.server_config.port} {config.server_config.server_dir}'
echo \$CMD
eval \$CMD
" > $server_log_file 2> $server_err_file &
SERVER_PID=$!

# Give server time to start and initialize
echo "Waiting for server to initialize..."
sleep 45
"""

    # Client setup and execution with redirected output
    # Calculate the total GPUs per node
    total_gpus_per_node = config.server_config.slurm.gpus_per_task * config.server_config.slurm.ntasks_per_node
    client_nodes = config.total_nodes - 1
    
    client_cmd = f"""
# Start clients on remaining nodes (all nodes except the first one)
client_nodes={client_nodes}
echo "Starting clients on $client_nodes nodes, logs will be written to $client_log_file"
set -x

srun --nodes=$client_nodes --ntasks=$client_nodes --relative=1 --environment={config.server_config.slurm.environment} bash -c "
echo 'Client starting on node: $(hostname)'
echo 'Using C10D head node: $client_head_node ($client_head_ip)'
echo 'Using Mixtera server: $head_node_ip:{config.server_config.port}'

pushd {config.client_config.mixtera_dir}
n=0

until [ \\$n -ge 5 ]
do
   if pip install -e .; then
      echo 'pip install succeeded after try ('\\$n')'
      break
   else
      n=\\$((\\$n+1))
      echo 'pip install failed, retrying ('\\$n')'
      sleep 1
   fi
done
if [ \\$n -ge 5 ]; then
   echo 'pip install failed after 5 retries'
   exit 1
fi

popd
echo 'Running client with torchrun:'
CMD='numactl --membind=0-3 torchrun --nnodes=$client_nodes --nproc_per_node={total_gpus_per_node} --rdzv_backend c10d --rdzv_endpoint $client_head_ip:29500 {config.client_config.torchtitan_src}/train.py --job.config_file {config.client_config.config_file} --mixtera.ip '$head_node_ip' --mixtera.port {config.server_config.port}'
echo \$CMD
eval \$CMD
" > $client_log_file 2> $client_err_file &
CLIENT_PID=$!

echo "Waiting for client processes to complete..."
wait $CLIENT_PID
CLIENT_EXIT_CODE=$?
echo "Client processes completed with exit code $CLIENT_EXIT_CODE, waiting 2 minutes before stopping server..."
JOB_EXIT_CODE=$CLIENT_EXIT_CODE

# Wait 2 minutes to allow the server to complete any final operations
sleep 120

# Check if server srun process is still running
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server is still running after 2-minute grace period, terminating..."
    
    # First, try to signal the server srun process
    kill $SERVER_PID
    
    # Directly cancel the server job step without using squeue
    # The main server process is likely running as step 0
    echo "Attempting to cancel server job step: $SLURM_JOB_ID.0"
    scancel $SLURM_JOB_ID.0
    
    # Give it a moment to shut down gracefully
    sleep 10
    
    # Force kill if still running
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server didn't terminate gracefully, force killing..."
        kill -9 $SERVER_PID
    fi
else
    echo "Server has already completed"
fi

echo "All processes completed, job exit code: $JOB_EXIT_CODE"
exit $JOB_EXIT_CODE
"""

    # Combine all parts
    sbatch_script = (
        sbatch_header
        + env_var_lines
        + additional_setup
        + node_setup
        + server_cmd
        + client_cmd
    )

    return sbatch_script


def main():
    parser = argparse.ArgumentParser(
        description="Run Mixtera server and TorchTitan client in a single job."
    )
    parser.add_argument("server_config_path", type=str, help="Path to the server configuration file.")
    parser.add_argument("client_config_path", type=str, help="Path to the client configuration file.")
    parser.add_argument("--chained_jobs", type=int, default=1, help="Number of consecutive jobs to run, each dependent on the successful completion of the previous one.")
    args = parser.parse_args()

    # Load the server configuration
    server_config = MixteraServerConfig.from_yaml(args.server_config_path)
    
    # Load the client configuration
    client_config = TorchtitanConfig.from_yaml(args.client_config_path)
    
    # Extract Slurm configuration from client's config file
    with open(args.client_config_path, "r") as f:
        client_data = yaml.safe_load(f)

    slurm_data = client_data.get("slurm", {})
    client_slurm_config = MixteraSlurmConfig(
        job_name=slurm_data["job_name"],
        log_dir=Path(client_config.torchtitan_logs),
        partition=slurm_data["partition"],
        ntasks_per_node=slurm_data["ntasks_per_node"],
        gpus_per_task=slurm_data["gpus_per_task"],
        time=slurm_data["time"],
        environment=slurm_data["environment"],
        account=slurm_data["account"],
        exclude=slurm_data.get("exclude")
    )
    
    # Determine total nodes needed - client nodes + 1 for server
    client_nodes = slurm_data["nodes"]
    total_nodes = client_nodes + 1
    
    # Create the combined config
    combined_config = CombinedConfig(
        server_config=server_config,
        client_config=client_config,
        total_nodes=total_nodes
    )
    
    # Set client's slurm config (needed for path references)
    setattr(client_config, "slurm", client_slurm_config)
    
    # Build the sbatch script
    sbatch_script = build_sbatch_script(combined_config)

    # Create a unique name for the sbatch file
    sbatch_file = f"run_combined_{client_config.job_name}_{client_config.run_ident}.sbatch"
    
    # Save the sbatch script
    with open(sbatch_file, "w") as f:
        f.write(sbatch_script)
    print(f"SBATCH script saved to: {sbatch_file}")
    print(f"Submitting chain of {args.chained_jobs} jobs")
    
    # Submit first job
    proc = subprocess.run(
        ["sbatch", sbatch_file],
        env=get_no_conda_env(),
        capture_output=True,
        text=True
    )
    
    if proc.returncode != 0:
        print(f"Error submitting first job: {proc.stderr}")
        return
    
    # Extract job ID from first submission
    try:
        job_id = proc.stdout.strip().split()[-1]
        print(f"Submitted job 1/{args.chained_jobs} with ID: {job_id}")
    except (IndexError, ValueError):
        print(f"Could not extract job ID from: {proc.stdout}")
        return
    
    # Submit remaining jobs in the chain
    prev_job_id = job_id
    for i in range(2, args.chained_jobs + 1):
        proc = subprocess.run(
            ["sbatch", f"--dependency=afterok:{prev_job_id}", sbatch_file],
            env=get_no_conda_env(),
            capture_output=True,
            text=True
        )
        
        if proc.returncode != 0:
            print(f"Error submitting job {i}: {proc.stderr}")
            break
        
        try:
            job_id = proc.stdout.strip().split()[-1]
            print(f"Submitted job {i}/{args.chained_jobs} with ID: {job_id}, dependent on job {prev_job_id}")
            prev_job_id = job_id
        except (IndexError, ValueError):
            print(f"Could not extract job ID from: {proc.stdout}")
            break



if __name__ == "__main__":
    main()
