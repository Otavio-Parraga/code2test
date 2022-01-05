import os
import tarfile
from pathlib import Path

data_path = Path('methods2test/corpus/raw/fm/')
all_files = os.listdir(data_path)

for file in all_files:
    print(f'Split {file}')
    file_name = data_path / file
    my_tar = tarfile.open(file_name, 'r:bz2')
    my_tar.extractall(data_path)
    my_tar.close()
