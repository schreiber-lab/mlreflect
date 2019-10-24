from typing import List


def make_label_names(number_of_layers: int) -> List[str]:
    label_names = ['' for i in range(number_of_layers * 3)]

    for layer_index in range(number_of_layers):
        label_names[layer_index] = f'layer{number_of_layers - layer_index}_thickness'
        label_names[
            layer_index + number_of_layers] = f'layer{number_of_layers - layer_index}_roughness'
        label_names[
            layer_index + 2 * number_of_layers] = f'layer{number_of_layers - layer_index}_sld'

    return label_names
