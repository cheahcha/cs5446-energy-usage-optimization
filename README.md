# Optimization of Energy Usage in Singapore


This project aims to forecast and optimize electricity consumption patterns in Singapore using historical data and time-series forecasting models.


## Getting Started
### Prerequisites
Install the required libraries.

```
pip install -r requirements.txt
```

## Running the Model
To run the model, execute the following command:

```
python main.py
```

## Files

### Main script
- **main.py**: Main script to run the forecasting model and MDP.
### Helper Classes
- constants.py: Contains the region labels and policy maps.
- Predictor.py: Contains the Predictor class used for final forecasting.
- MDP.py: Contains the MDP class used for constructing transition and reward matrices.

### Data Directory
- data/: Directory containing the raw and processed energy consumption data.

### EDA + Forecasting
- forecasting_eda.ipynb: Contains the code cells for plotting time-series and generating forecasts.
- utils/: Contains forecasting and plotting functions for forecasting_eda.ipynb

### Evaluation
- mdp_evaluation.ipynb: Contains the code cells for comparing original consumption data against the electricity consumption with the MDP-based model.
