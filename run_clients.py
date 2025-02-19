import argparse
import os
import subprocess
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class SlurmConfig:
    job_name: str
    time: str
    partition: str
    account: Optional[str]
    ntasks_per_node: int
    gpus_per_task: int
    nodes: int
    exclude: Optional[str]
    environment: str


@dataclass
class TorchtitanConfig:
    slurm: SlurmConfig
    torchtitan_src: Path
    torchtitan_logs: Path
    config_file: Path  # Path to the torchtitan TOML config
    environment_vars: dict
    additional_setup: Optional[Path]
    mixtera_server_config: Path
    run_ident: str
    job_name: str
    mixtera_dir: Path
    mixtera_ip: Optional[str] = None  # To be filled from mixtera server config
    mixtera_port: Optional[int] = None  # To be filled from mixtera server config

    @classmethod
    def from_yaml(cls, path: str) -> "TorchtitanConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Parse the 'slurm' section
        slurm_data = data.get("slurm", {})
        slurm_config = SlurmConfig(
            job_name=slurm_data["job_name"],
            time=slurm_data["time"],
            partition=slurm_data["partition"],
            account=slurm_data.get("account"),
            ntasks_per_node=slurm_data["ntasks_per_node"],
            gpus_per_task=slurm_data["gpus_per_task"],
            nodes=slurm_data["nodes"],
            exclude=slurm_data.get("exclude"),
            environment=slurm_data["environment"],
        )

        # Create the torchtitan config
        torchtitan_config = cls(
            slurm=slurm_config,
            torchtitan_src=Path(data["torchtitan_src"]),
            torchtitan_logs=Path(data["torchtitan_logs"]),
            config_file=Path(data["config_file"]),
            environment_vars=data.get("environment_vars", {}),
            additional_setup=Path(data["additional_setup"])
            if data.get("additional_setup")
            else None,
            mixtera_server_config=Path(data["mixtera_server_config"]),
            run_ident=data.get("run_ident", "run"),
            job_name=data.get("job_name", slurm_data["job_name"]),
            mixtera_dir=Path(data["mixtera_dir"]),
        )

        return torchtitan_config

    def load_mixtera_server_info(self):
        with open(self.mixtera_server_config, "r") as f:
            mixtera_server_data = yaml.safe_load(f)

        # Get the port
        self.mixtera_port = mixtera_server_data["port"]

        # Get the server_ip.txt path from mixtera_server_data['slurm']['log_dir']
        mixtera_slurm_log_dir = Path(mixtera_server_data["slurm"]["log_dir"])
        server_ip_file = (
            mixtera_slurm_log_dir
            / f'server_ip_{mixtera_server_data['slurm']['job_name']}.txt'
        )

        # Read the server IP from the file
        if server_ip_file.exists():
            with open(server_ip_file, "r") as ip_file:
                self.mixtera_ip = ip_file.read().strip()
        else:
            raise FileNotFoundError(
                f"Could not find Mixtera server IP file at {server_ip_file}"
            )


def get_no_conda_env():
    env = os.environ.copy()

    conda_prefix = env.get("CONDA_PREFIX", "")
    conda_bin = os.path.join(conda_prefix, "bin")

    keys_to_remove = [key for key in env if "CONDA" in key or "PYTHON" in key]
    for key in keys_to_remove:
        del env[key]

    paths = env["PATH"].split(os.pathsep)
    paths = [p for p in paths if conda_bin not in p and conda_prefix not in p]
    env["PATH"] = os.pathsep.join(paths)

    return env


def build_sbatch_script(config: TorchtitanConfig, log_dir: Path) -> str:
    output_file = log_dir / "output.log"
    error_file = log_dir / "output.err"

    # Build the SBATCH header
    sbatch_header = f"""#!/bin/bash
#SBATCH --job-name={config.slurm.job_name}
#SBATCH --nodes={config.slurm.nodes}
#SBATCH --ntasks-per-node={config.slurm.ntasks_per_node}
#SBATCH --time={config.slurm.time}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --gpus-per-task={config.slurm.gpus_per_task}
#SBATCH --account=A-a09
"""

    if config.slurm.account:
        sbatch_header += f"#SBATCH --account={config.slurm.account}\n"

    if config.slurm.partition:
        sbatch_header += f"#SBATCH --partition={config.slurm.partition}\n"

    if config.slurm.exclude:
        sbatch_header += f"#SBATCH --exclude={config.slurm.exclude}\n"

    # Environment variable definitions
    env_var_lines = "\n"
    for var_name, value in config.environment_vars.items():
        env_var_lines += f"export {var_name}={value}\n"

    # Additional setup commands if any
    additional_setup = ""
    if config.additional_setup and config.additional_setup.exists():
        with open(config.additional_setup, "r") as f:
            additional_setup = f.read()

    # Build the commands to get the head node IP (taken from torchtitan repo)
    sbatch_body = """
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
"""

    # Build the torchrun command
    total_gpus_per_node = config.slurm.gpus_per_task * config.slurm.ntasks_per_node
    torchrun_cmd = f"""
srun -ul --container-writable --environment={config.slurm.environment} bash -c "
pushd {config.mixtera_dir}
n=0

until [ \$n -ge 5 ]
do
   if pip install -e .; then
      echo 'pip install succeeded after try ('\$n')'
      break
   else
      n=\$((\$n+1))
      echo 'pip install failed, retrying ('\$n')'
      sleep 1
   fi
done
if [ \$n -ge 5 ]; then
   echo 'pip install failed after 5 retries'
   exit 1
fi

popd
numactl --membind=0-3 torchrun --nnodes={config.slurm.nodes} --nproc_per_node={total_gpus_per_node} --rdzv_backend c10d --rdzv_endpoint '$head_node_ip:29500' {config.torchtitan_src}/train.py --job.config_file {config.config_file} --mixtera.ip {config.mixtera_ip} --mixtera.port {config.mixtera_port}
"
"""

    sbatch_script = (
        sbatch_header
        + env_var_lines
        + "\n"
        + additional_setup
        + "\n"
        + sbatch_body
        + "\n"
        + torchrun_cmd
    )

    return sbatch_script


def main():
    parser = argparse.ArgumentParser(
        description="Torchtitan training launcher with Mixtera."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the launcher configuration file."
    )
    args = parser.parse_args()

    # Load the configuration
    config = TorchtitanConfig.from_yaml(args.config_path)

    # Load Mixtera server IP and port
    config.load_mixtera_server_info()

    # Prepare the log directory
    log_ident = f"{config.run_ident}_{config.job_name}"
    log_dir = config.torchtitan_logs / config.job_name / log_ident
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory created: {log_dir}")

    # Build the sbatch script
    sbatch_script = build_sbatch_script(config, log_dir)

    # Save sbatch script
    sbatch_file = log_dir / "run_torchtitan.sbatch"
    with open(sbatch_file, "w") as f:
        f.write(sbatch_script)
    print(f"SBATCH script saved to: {sbatch_file}")

    # Submit the sbatch script
    print(f"Submitting job for {sbatch_file}")

    proc = subprocess.run(
        ["sbatch", sbatch_file], capture_output=True, text=True, env=get_no_conda_env()
    )

    print(f"Job submission output:\n{proc.stdout}")


if __name__ == "__main__":
    main()
