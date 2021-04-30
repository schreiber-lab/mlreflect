def _build_from_dict(experiment):
    pass
    # remap


def load_json(path):
    import json
    with open(path, 'r') as fid:
        data = fid.read()
    experiment = json.loads(data)
    return _build_from_dict(experiment)
