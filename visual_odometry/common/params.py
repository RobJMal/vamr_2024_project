import yaml
from typing import Any

class ParamServer:
    def __init__(self, config_file):
        self._config_file = config_file
        with open(config_file, 'r') as f:
            self._params = yaml.load(f, Loader=yaml.FullLoader)

    def __getitem__(self, keys) -> Any:
        """Retrieve nested keys using a tuple of keys.
        
        :param keys: Tuple of keys to retrieve from the configuration.
        :type keys: tuple

        :return: Value of the nested key.
        :rtype: Any
        """
        result = self._params
        for key in keys if isinstance(keys, tuple) else (keys,):
            try:
                result = result[key]
            except KeyError:
                raise KeyError(f"Key '{key}' not found in configuration.")
        return result
    
    def __str__(self):
        return f"ParamServer({self._config_file})"

# Simple test case
if __name__ == "__main__":
    param_server = ParamServer("params/pipeline_params.yaml")
    print(param_server[("keypoint_tracking", "param_1")])