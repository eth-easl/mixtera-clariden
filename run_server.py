import argparse
import os
import yaml
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

    @classmethod
    def from_yaml(cls, path: str) -> 'MixteraServerConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse the 'slurm' section
        slurm_data = data.get('slurm', {})
        slurm_config = MixteraSlurmConfig(
            job_name=slurm_data['job_name'],
            log_dir=Path(slurm_data['log_dir']),
            partition=slurm_data['partition'],
            ntasks_per_node=slurm_data['ntasks_per_node'],
            gpus_per_task=slurm_data['gpus_per_task'],
            time=slurm_data['time'],
            cpus_per_task=slurm_data['cpus_per_task'],
            environment=slurm_data['environment']
        )

        # Create the server config
        server_config = cls(
            slurm=slurm_config,
            server_dir=Path(data['server_dir']),
            mixtera_dir=Path(data['mixtera_dir']),
            port=data['port']
        )
        return server_config

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