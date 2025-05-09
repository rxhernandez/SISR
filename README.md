# Stochiometrically Informed Symbolic Regression (SISR)


----------------

## Description
----------------

A stoichiometrically-informed method to fit the rate constants in a reaction mechanism through differential optimization and couple that fitting method with a genetic optimization approach that searches a symbolic space of possible reaction mechanisms to find the mechanism that best matches a time-series dataset of concentrations.
It returns a symbolic chemical reaction mechanism, the rate constant for each reaction in the mechanism, and the kinetic equations that describe the chemical process under examination.

## Contents
----------------

This repository includes the codes for the paper : **[Stoichiometrically-informed symbolic regression for extracting chemical reaction mechanisms from data](link)** (Manuel Palma Banos,Joel D. Kress,Rigoberto Hernandez,Galen T. Craven,Journal Info, Year).

and the following content:

Example input datasets and scripts to run the SISR code: examples/

Extact results included in our paper: results/

Paper and associated figures: docs/

The python3 based source code: src/

## How to use
----------------

### Installation
```bash
git clone https://github.com/rxhernandez/SISR
cd SISR
./install.sh
```
If system was intalled correctly and unittests passed, you should see the following message:
```
Ran 19 tests in X.XXXs

OK
```


### Verification
```bash
source python_venvs/SISR/bin/activate
pip list | grep SISR
```
You should see the installed SISR package listed as follows:
```
SISR                          0.0.1
```

### Example
```bash
cd examples
python LK_example.py
```

### Uninstall
```bash
deactivate
./uninstall.sh
```
<hr>

Citing
----------------

If you use database or codes, please consider citing the paper:

>M. Palma Banos, J. D. Kress, R. Hernandez, G. T. Craven, "Stoichiometrically-informed symbolic regression for ex-
tracting chemical reaction mechanisms from data," (in preparation).

and/or this site:

>M. Palma Banos, J. D. Kress, R. Hernandez, G. T. Craven, SISR, URL, [https://github.com/rxhernandez/SISR](https://github.com/rxhernandez/SISR)

<hr>

Acknowledgment
----------------

This work was supported by the Los Alamos National Laboratory (LANL) Directed Research and Development funds (LDRD).
This work was supported by the National Science Foundation through Grant No.~CHE 2102455.
<hr>

License
----------------

SISR code is distributed under terms of the [MIT License]
