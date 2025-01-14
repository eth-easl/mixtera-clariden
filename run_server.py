import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass
class MixteraSlurmConfig:
    job_name: str
    log_dir: Path
    partition: Literal["normal", "debug"]
    ntasks_per_node: int
    gpus_per_task: int
    time: str

@dataclass
class MixteraServerConfig:
    slurm: MixteraSlurmConfig
    server_dir: Path
    mixtera_dir: Path
    port: int

def run_mixtera_server(config: MixteraServerConfig) -> None:
    sbatch_header = f"""#!/bin/bash
#SBATCH --job-name={config.slurm.job_name}
#SBATCH --output={config.slurm.log_dir}/%j.out
#SBATCH --environment={config.slurm.environment}
#SBATCH --partition={config.slurm.partition}
#SBATCH --ntasks-per-node={config.slurm.ntasks_per_node}
#SBATCH --cpus-per-task={config.slurm.cpus_per_task}
#SBATCH --time={config.slurm.time}\n
"""
    
    server_setup_cmd = "\nexport SERVER_IP=$(hostname)\n"
    server_setup_cmd += f"export SERVER_DIR={config.server_dir}\n"
    server_setup_cmd += f"export SERVER_PORT={config.port}\n"
    server_setup_cmd += f"\necho $SERVER_IP > {config.slurm.log_dir}/server_ip.txt\n"
    server_setup_cmd += f"\npushd {config.mixtera_dir} && pip install -e . && popd"

    # No need to call srun here because we only need a single node for the server
    server_setup_cmd += r"""
numactl --membind=0-3 python -u -m mixtera.network.server.entrypoint $SERVER_DIR \
    --host $SERVER_IP \
    --port $SERVER_PORT
""" 

    sbatch_script = sbatch_header + server_setup_cmd

    sbatch_file = "run_mixtera_server.sbatch"

    with open(sbatch_file, "w") as f:
        f.write(sbatch_script)

    cmd = f"sbatch {sbatch_file}"
    os.popen(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mixtera server with specified config.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    config = MixteraServerConfig.from_yaml(args.config_path)
    run_mixtera_server(config)