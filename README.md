# Stochiometrically Informed Symbolic Regression (SISR)

[License?]

Code for the paper: **[Stoichiometrically-informed symbolic regression for extracting chemical reaction mechanisms from data](link)** (Manuel Palma Banos,Joel D. Kress,Rigoberto Hernandez,Galen T. Craven,Journal Info, Year).

## Description
A stoichiometrically-informed method to fit the rate constants in a reaction mechanism through differential optimization and couple that fitting method with a genetic optimization approach that searches a symbolic space of possible reaction mechanisms to find the mechanism that best matches a time-series dataset of concentrations.
It returns a symbolic chemical reaction mechanism, the rate constant for each reaction in the mechanism, and the kinetic equations that describe the chemical process under examination.

## Installation
```bash
git clone https://github.com/...
cd ...
pip install -r requirements.txt