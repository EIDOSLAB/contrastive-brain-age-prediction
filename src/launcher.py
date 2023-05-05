import yaml
import sys
import subprocess
import os
from yaml.loader import SafeLoader


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage: ./launcher.py path/to/yaml")
        exit(1)

    with open(sys.argv[1]) as f:
        data = yaml.load(f, Loader=SafeLoader)
    
    program = data['program']
    del data['program']

    skip = False
    for idx, override in enumerate(sys.argv[2:]):
        if skip: 
            skip = False
            continue
        
        if '=' in override:
            k, v = override.split('=')
        else:
            k = override.replace('--', '')
            v = sys.argv[2+idx+1]
            skip = True
        data[k] = v

    args = ["python3", os.path.join(os.getcwd(), program)]
    for k, v in data.items():
        args.extend(["--" + k, str(v)])
    print("Running:", ' '.join(args))
    subprocess.run(args)
