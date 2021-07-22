import pandas as pd
from glob import glob
from pathlib import Path
from collections import OrderedDict

from ..p08tools import read_fio
from ..p08tools import Scan


class FioParser:
    """Parse data from .fio files and their corresponding detector image files within a folder

    Args:
        file_stem: The name of the experiment including the preceding folder structure. E.g. if you're experiment
            name is ``'my_data'`` and in the folder ``'user/data/'``, the file stem would be ``'user/data/my_data'``.
            This will look for all scans in the folder ``'user/data/'`` that begin with ``'my_data'``.
        theta_name: Name of the counter that contains half the scattering angle (default: ``'om'``).
        two_theta_name: Name of the counter that contains the full scattering angle (default: ``'tt'``).
    """
    def __init__(self, file_stem: str, theta_name: str = 'om', two_theta_name: str = 'tt'):
        self._theta_name = theta_name
        self._two_theta_name = two_theta_name

        self.file_stem = Path(file_stem)
        print(f'found {self.number_of_scans} scans for {str(self.file_stem)}')

    @property
    def scan_info(self):
        scan_info_dict = OrderedDict()
        for idx, file_name in zip(self.scan_numbers, self.fio_path_list):
            header = read_fio(file_name)[3]
            is_theta2theta_scan = self._is_theta2theta_scan(header['scan_cmd'])

            scan_info_dict[idx] = {
                'file_name': file_name,
                'header': header,
                'is_theta2theta_scan': is_theta2theta_scan
            }
        return scan_info_dict

    @property
    def fio_path_list(self):
        search_path = str(self.file_stem) + '*.fio'
        return sorted(glob(search_path))

    @property
    def number_of_scans(self):
        return len(self.scan_info)

    @property
    def scan_numbers(self):
        return [int(file.split(str(self.file_stem) + '_')[1].strip('.fio')) for file in self.fio_path_list]

    def parse_fio(self, scan_number: int) -> dict:
        """Extract the data of a given scan (including header) into a dictionary."""
        motor_positions, column_names, counters, header_info = read_fio(str(self.file_stem) + f'_{scan_number:05d}.fio')

        output = {
            'header_info': header_info,
            'motor_positions': motor_positions,
            'counters': pd.DataFrame(data=counters)
        }
        return output

    def extract_scan(self, scan_number: int) -> Scan:
        """Extract the data of the entire image stack for the given scan into a ``Scan`` object."""
        scan = Scan()
        scan.load_scan(str(self.file_stem) + f'_{scan_number:05d}.fio')
        scan.is_theta2theta_scan = self._is_theta2theta_scan(scan.scan_cmd)
        return scan

    def _is_theta2theta_scan(self, command):
        return self._theta_name in command and self._two_theta_name in command
