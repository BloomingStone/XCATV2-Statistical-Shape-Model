from pathlib import Path


def parse_parameters(param_path: Path) -> dict[str, int|float|str]:
    params = {}
    with open(param_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            key, value = line.split('=')[:2]
            key = key.strip()
            value = value.split('#')[0].strip()
            
            if '.' in value:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
            elif value.isdigit():
                params[key] = int(value)
            else:
                params[key] = value

    return params

def test():
    root_dir = Path.cwd().resolve()
    param_path = root_dir / "parameters/general.samp.par"
    assert param_path.exists()
    params = parse_parameters(param_path)
    assert params['mode'] == 0
    assert params['heart_base'] == "vmale50_heart.nrb"
    for key, value in params.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test()