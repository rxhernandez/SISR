# Stochiometrically Informed Symbolic Regression (SISR)


----------------

This repository includes the following content:
Training and validation datasets in folders: simulation/ and TEM/.
Testing unknown datasets in folders: predictVMD/ and predictTEM/.
The python3 codes are in the folder: DNA_origami_classification_MLAI/.
All the 10 different runs are in the folder: results_plots/ with the python code to do data analysis and plot.

Details on the use and installation of the code is available in the [user_guide.md](https://github.com/rxhernandez/TEMCOIN/blob/main/user_guide.md).

Code for the paper: **[Stoichiometrically-informed symbolic regression for extracting chemical reaction mechanisms from data](link)** (Manuel Palma Banos,Joel D. Kress,Rigoberto Hernandez,Galen T. Craven,Journal Info, Year).

## Description
A stoichiometrically-informed method to fit the rate constants in a reaction mechanism through differential optimization and couple that fitting method with a genetic optimization approach that searches a symbolic space of possible reaction mechanisms to find the mechanism that best matches a time-series dataset of concentrations.
It returns a symbolic chemical reaction mechanism, the rate constant for each reaction in the mechanism, and the kinetic equations that describe the chemical process under examination.

## Installation
```bash
git clone https://github.com/...
cd ...
pip install -r requirements.txt
(end)
```

<hr>

Citing
----------------

If you use database or codes, please consider citing the paper:

>X. Wei, Q. Mo, C. Chen, M. Bathe and R. Hernandez, "DNA Origami Nanostructures Observed in Transmission Electron Microscopy Images can be Characterized through Convolutional Neural Networks," (in preparation).

and/or this site:

>X. Wei, Q. Mo and R. Hernandez, TEMCOIN, URL, [https://github.com/rxhernandez/TEMCOIN](https://github.com/rxhernandez/TEMCOIN)

<hr>

Acknowledgment
----------------

This work was supported by the National Science Foundation through Grant No.~CHE 2102455.


<hr>

License
----------------

NestedAE code and databases are distributed under terms of the [MIT License](https://github.com/rxhernandez/TEMCOIN/blob/main/LICENSE).


