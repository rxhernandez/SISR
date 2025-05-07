python3.11 -m venv python_venvs/SISR
source python_venvs/SISR/bin/activate
pip install --upgrade pip setuptools wheel
pip install .
python -m unittest discover -s tests -v
