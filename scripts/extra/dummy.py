import argparse
from scripts.core_utils import create_paths_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script trains the model'
    )
    parser.add_argument('--path',
                        help='path',
                        default="./data-split/val",
                        type=str)

    args = parser.parse_args()

create_paths_list(args.__dict__)