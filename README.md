# TTT4180 - Technical Acoustics - Assignement 0

## Structure
All the relevant parts for the code are placed in `Code/`, while the 
files used for generating the report are in `Reports/`.

## Code
To run the code first setup the Python environment, e.g. with (in Linux):
1. `cd Code/`
2. `python3 -m venv venv`
3. `source venv/bin/activate`
4. `pip install -r requirements.txt`
5. `python assignment_0.py --help`
6. `python assignment_0.py --input path_to_input_file`

`assignment_0.py` is only used to provide a CLI for the code in `compute_values.py`.
The option `--all` does not work if the files with the hard-coded paths are not
available.

