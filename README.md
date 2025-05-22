# CIFO - Project
Computational Intelligence for Optimization Project

## Project Overview
This repository contains the materials for our project in the CIFO course, part of the MDSAA @NOVAIMS. The project focuses on apply Genetic Algorithms (GA) to solve a complex scheduling problem: the **Music Festival Lineup Optimization**. The objective is to design an optimal festival schedule by assigning artists to stages and time slots in a way that maximizes artist popularity during prime time, ensures genre diversity, and minimizes fan base conflicts.

## Project Structure

```text
CIFO2025/
├── data/
│   ├── artists(in).csv        # Artists dataset: Name, Genre and Popularity Score
│   ├── conflicts(in).csv      # Conflict matrix (35x35) of conflict scores between each pair of artists
│   └── df_load.py             # script to load the data
├── library/                   # Core GA implementation modules
│   ├── algorithms.py
│   ├── solution.py
│   └── selection_and_operators/
│       ├── crossover.py
│       ├── mutation.py
│       └── selection.py
├── EDA.ipynb
├── MFLUSolution.py
├── experiment.py
├── results.ipynb
├── CIFO_2024_2025_Project_Statement.pdf  # Project requirements & guidelines
└── README.md
```

## Team Members
- Ana Marta Azinheira
- Carolina Pinto
- Catarina Ribeirinha
- José Cavaco

