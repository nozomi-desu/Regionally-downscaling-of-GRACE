"""Create runnable configurations without modifying the frozen original configurations."""
from pathlib import Path
import argparse
import yaml

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=root/'configs/paper_local')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = '/home/user02/grace_remote_train/'
    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, str) and value.startswith(prefix):
            return str(root/value[len(prefix):])
        return value
    for paper, project in [('M0','M0'),('M1','M1'),('M2','M2'),('M3','M3'),('M4','M5')]:
        original = root/f'data_driven_alpha_beta/{project}/config.yaml'
        config = relocate(yaml.safe_load(original.read_text(encoding='utf-8')))
        path = args.output_dir/f'{paper}.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        print(f'{paper} = project {project}: {path}')

if __name__ == '__main__':
    main()
