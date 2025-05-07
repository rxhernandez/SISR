python3.11 -m venv python_venvs/SISR
source python_venvs/SISR/bin/activate
pip install -U build
python -m build --wheel
pip install .
python -m unittest discover -s tests -v
