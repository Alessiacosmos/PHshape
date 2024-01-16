# PH-shape
Code repository of [PH-shape](https://doi.org/10.1080/10095020.2023.2280569).  
PH-shape achieves the outline extraction of building footprints from point clouds, with adaptive parameters. 


## Usage
use through IDE
by running `main_codes_gudhi/main_all_gu.py`.  

or use through terminal
```shell
python3 -m main_codes_gudhi.main_all_gu --config config/trd/config_trd_gu_400_terminal.yaml
```

The following requirements are necessary for PH-shape:
- gudhi
- shapely

Other potential requirments can be found in [requirements.txt](requirements.txt) or according to the compilation errors.

## Data
The test data in Trondheim, Norway can be downloaded [here](https://drive.google.com/drive/folders/1K3DhWqzkXhoFQRaxyjnm4UpUoR1gsasH?usp=sharing).

## License
The code is licensed under the [MIT License](LICENSE)
