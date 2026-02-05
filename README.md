# Epidemiology-Informed Spatio-Temporal Graph Neural Network for Dengue Prediction

An epidemiology-informed spatio-temporal graph neural network model for predicting dengue fever cases in Taiwan.

## Model Architecture

### Core Components

1. **Spatial Modeling**: Graph-based spatial attention mechanism to capture inter-city spatial correlations
2. **Temporal Modeling**: Multi-scale temporal encoder with positional encoding and LSTM for temporal pattern learning
3. **SIS Regularization**: SIS (Susceptible-Infected-Susceptible) dynamics model as a soft constraint

### Key Features

- **Hybrid Modeling**: Combines data-driven learning with epidemiological priors
- **Multi-Scale Temporal Processing**: Captures patterns at different time scales (3, 7, 14, 30 days)
- **Attention Mechanisms**: Both spatial and temporal attention for improved feature extraction
- **Soft Constraints**: Epidemiological dynamics guide learning without hard constraints
- **Interpretability**: Explicit disease dynamics modeling
- **Extensibility**: Supports learnable SIS parameters, city-specific parameters, and mobility information

## Project Structure

```
.
├── Dengue_Daily_EN.csv          # Raw data
├── data_preprocessing.py         # Data preprocessing script
├── model.py                      # Model definition (EpidemiologyGNNv2)
├── train.py                      # Training script with advanced loss functions
├── inference.py                  # Single/multi-day prediction
├── visualize_results.py          # Result visualization
├── comprehensive_evaluation.py   # Comprehensive model evaluation
├── multi_step_evaluation.py      # Multi-step forecasting evaluation
├── run_pipeline.py               # One-click pipeline runner
└── README.md                     # Documentation
```

## Quick Start

### Option 1: One-Click Pipeline

```bash
python run_pipeline.py
```

This will automatically execute:
1. Data preprocessing
2. Model training
3. Result visualization

### Option 2: Step-by-Step Execution

#### Step 1: Data Preprocessing

```bash
python data_preprocessing.py
```

This will:
- Load raw CSV data
- Aggregate cases by city and date
- Create time series samples using sliding windows
- Split into train/validation/test sets (2000-2020 for training, 2020-2024 for testing)
- Save preprocessed data to `processed_data.pkl`

#### Step 2: Train Model

```bash
python train.py
```

The training process will:
- Load preprocessed data
- Create fully-connected graph structure
- Initialize the EpidemiologyGNNv2 model
- Train with combined loss (MSE + MAE + Outbreak-weighted + SIS consistency)
- Use mixed precision training (AMP) for efficiency
- Apply early stopping and cosine annealing learning rate schedule
- Save best model and training curves

#### Step 3: Visualize Results

```bash
python visualize_results.py
```

This generates:
- MAE/RMSE metrics by city for different forecast horizons (3, 7, 14, 30 days)
- Time series prediction plots for each city
- Horizon comparison plots (2x2 grid for each city)
- All outputs are saved to `visualization_results/`

#### Step 4: Run Inference

```bash
# Predict using test data for 7 days ahead
python inference.py --checkpoint checkpoints/best_model.pth --days 7 --use_test_data
```

## Model Configuration

You can modify the following configurations in `train.py`:

```python
config = {
    'batch_size': 512,
    'learning_rate': 5e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-5,
    'num_epochs': 300,
    'patience': 50,              # Early stopping patience
    'grad_clip': 1.0,
    
    # Loss weights
    'lambda_mae': 0.5,
    'lambda_outbreak': 0.3,
    'lambda_sis': 0.05,
    
    # Model architecture
    'spatial_hidden_dim': 128,
    'temporal_hidden_dim': 256,
    'num_spatial_layers': 3,
    'num_temporal_layers': 3,
    'dropout': 0.15,
    'use_sis': True,
    'use_amp': True,             # Mixed precision training
}
```

## Model Architecture Details

### Input/Output

- **Input**: `X ∈ R^(B×w×N)` where B is batch size, w is window size (14 days default), N is number of cities
- **Output**: `Î_{t+1} ∈ R^N`, predicted case counts for the next time step

### Loss Function

Total Loss = MSE Loss + λ_mae × MAE Loss + λ_outbreak × Outbreak Loss + λ_sis × SIS Consistency Loss

```
L = L_MSE + λ_mae × L_MAE + λ_outbreak × L_outbreak + λ_sis × L_SIS
```

Where:
- `L_MSE`: Mean Squared Error on log-transformed predictions
- `L_MAE`: Mean Absolute Error for robustness
- `L_outbreak`: Weighted loss emphasizing outbreak periods
- `L_SIS`: Consistency with SIS dynamics model

### SIS Dynamics Model

```
I_{t+1}^{SIS} = I_t + β(N - I_t)I_t/N - γI_t
```

Where:
- `I_t`: Current infection ratio
- `β`: Infection rate (learnable)
- `γ`: Recovery rate (learnable)
- `N`: Total population (normalized to 1)

## Output Files

After training, the following files are generated:

- `processed_data.pkl`: Preprocessed data
- `checkpoints/best_model.pth`: Best model weights with scaler information
- `checkpoints/test_results.pkl`: Test set results
- `checkpoints/training_curves.png`: Training curves
- `visualization_results/`: Visualization outputs
  - `fig1_*.png`: MAE/RMSE metrics by city and horizon
  - `city_predictions/`: Per-city prediction plots

## Evaluation Metrics

The model is evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

## Future Directions

1. **Distance-Weighted Graphs**: Use geographic distance for weighted graph construction
2. **Mobility Graphs**: Incorporate population flow data
3. **Multi-Compartment Models**: Extend to SEIR and other complex models
4. **City-Specific Parameters**: Learn different SIS parameters per city
5. **External Features**: Include weather, temperature, and other environmental factors
6. **Uncertainty Quantification**: Add prediction intervals

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

## License

MIT License
