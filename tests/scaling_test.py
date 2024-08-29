import torch
import os
import argparse
import sys

# Make ../src visible for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import networks

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="APTS", required=True)
    parser.add_argument("--dataset", type=str, default="mnist", required=True)
    parser.add_argument("--batch_size", type=int, default=32, required=True)
    parser.add_argument("--model", type=str,
                        default="feedforward", required=True)
    parser.add_argument("--num_subdomains", type=int, default=1, required=True)
    parser.add_argument("--num_replicas_per_subdomain",
                        type=int, default=1, required=True)
    parser.add_argument("--num_stages_per_replica",
                        type=int, default=1, required=True)
    parser.add_argument("--epochs", type=int, default=10, required=True)
    parser.add_argument("--trial", type=int, default=1, required=True)
    return parser.parse_args(args)


def main(args, rank=None, master_addr=None, master_port=None, world_size=None):
    parsed_args = parse_args(args)
    # utils.prepare_distributed_environment(
    #     rank, master_addr, master_port, world_size, is_cuda_enabled=True)
    # utils.check_gpus_per_rank()

    # Make seed depend on trial number to ensure reproducibility. Make it something complicated.
    seed = 123456789 * parsed_args.trial
    torch.manual_seed(seed)

    nn = networks.construct_stage_list(
        parsed_args.model, parsed_args.num_stages_per_replica)

    print(parsed_args)
    print("\n")


if __name__ == '__main__':
    main(sys.argv[1:])
