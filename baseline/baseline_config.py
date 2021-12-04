import sys
import yaml
import argparse


def load_yaml_config(path, skip_lines=0):
    with open(path, 'r') as infile:
        for i in range(skip_lines):
            # Skip some lines (e.g., namespace at the first line)
            _ = infile.readline()

        return yaml.safe_load(infile)


def save_yaml_config(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def get_args():
    parser = argparse.ArgumentParser()

    ##### General settings #####
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed')

    ##### Dataset settings #####
    parser.add_argument('--num_groups',
                        type=int,
                        default=2,
                        help='Number of groups')

    parser.add_argument('--num_subjects_per_group',
                        type=int,
                        default=5,
                        help='Number of subjects in each group')

    parser.add_argument('--num_samples',
                        type=int,
                        default=60,
                        help='Number of sample size for each subject') 
                        
    parser.add_argument('--num_variables',
                        type=int,
                        default=5,
                        help='Number of observed variables')
    
    parser.add_argument('--max_lag',
                        type=int,
                        default=1,
                        help='Number of maximal time lag')


    ##### Other settings #####

    return parser.parse_args(args=sys.argv[1:])
