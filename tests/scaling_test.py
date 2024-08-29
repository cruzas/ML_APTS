import argparse
import sys


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_subdomains", type=int, required=True)
    parser.add_argument("--num_replicas_per_subdomain",
                        type=int, required=True)
    parser.add_argument("--num_stages_per_replica", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    return parser.parse_args(args)


def main(args):
    parsed_args = parse_args(args)
    print(parsed_args)
    print("\n")


if __name__ == '__main__':
    main(sys.argv[1:])
