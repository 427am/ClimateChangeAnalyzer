# Climage Change Analyzer

This project analyzes climate data, implementing predictive algorithms such as SARIMA and linear regression, and then visualizes the results.

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
    - Windows: `venv\scripts\activate`
    - macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

### Usage

Run the main script:

`python src/main.py`

Or use the command-line interface

`python src/cli.py --data data/climate_data.csv --action predict`

### Running Tests

First run this to find all the tests:
`python -m unittest discover tests`

Afterwards, tests can be run individually:
1. For data processing: `python -m unittest tests.test_data_processor`
2. For the algorithms: `python -m unittest tests.test_algorithms`
3. For the visualizer: `python -m unittest tests.test_visualizer`

## Project Structure
- `src/`: Source code
    - `algorithms.py`: Implementation of SARIMA and linear machine learning algorithms
    - `cli.py`: Command-line interface
    - `data_processor.py`: Handles data processing and cleanup
    - `main.py`: Main script
    - `visualizer.py`: Visualizes generated data into charts
- `tests/`: Unit tests
    - `__init__.py`: Init file
    - `test_algorithms.py`: Unit test for algorithms.py
    - `test_data_processor.py`: Unit test for data_processor.py
    - `test_visualizer.py`: Unit test for visualizer.py
- `data/`: Climate data (CSV format)
- `requirements.txt`: Project dependencies
- `projectreport_mla.pdf`: Project report in MLA formatting
- `projectreport_ieee.pdf`: Project report in IEEE formatting

## Project Features

Includes two machine learning models (SARIMA and linear-based) for predicting trends. Further explanation and results of the project can be found in the project report.

## Contributions

- Alexander Kajda: Implemented the SARIMA and linear model algorithms and their related unit tests for the project.

- Kaitlyn Franklin: Implemented the visualizers and related unit test for the project.

- Maddy Burns: Implemented the data processor, its related unit test, and the command-line interface for the project.

- Ömer Tüzün: Wrote the project report and the README.md for the project. Suggested the SARIMA model be used.
