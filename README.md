# NOx Conversion Efficiency Prediction System

**Advanced Machine Learning System for SCR (Selective Catalytic Reduction) Performance Prediction**

This project provides an end-to-end solution for predicting NOx conversion efficiency in diesel engine aftertreatment systems using machine learning.

## Dataset Overview

- **35 Features**: Complete engine, emissions, and environmental parameters
- **5,000 Samples**: Realistic synthetic data based on actual engine data
- **Target**: NOx Conversion Efficiency (%) prediction
- **Key Features**: Engine speed, torque, SCR temperatures, DEF injection rates, NOx concentrations
- **Data Location**: `data/sample_data.xlsx`

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment (recommended)
python -m venv nox_env
source nox_env/bin/activate  # Windows: nox_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python src/train_model.py
```
**Expected output:**
- Training time: 30-60 seconds
- R^2 Score: >0.90 (excellent performance)
- Model artifacts saved to `models/` directory

### 3. Launch GUI Predictor
```bash
python src/gui.py
```

## GUI Features

### **Simplified Input Interface**
The GUI only asks for **8 most important parameters** instead of all 35:

1. **NOx Inlet Concentration (ppm)** - Measurable upstream NOx
2. **Engine Speed (RPM)** - Primary operating point
3. **Engine Torque (Nm)** - Load indicator
4. **SCR Bed Temperature (°C)** - Critical for catalyst performance
5. **DEF Injection Rate (L/min)** - Controllable urea dosing
6. **Exhaust Flow Rate (kg/h)** - System flow parameter
7. **DOC Inlet Temperature (°C)** - Upstream temperature
8. **Ambient Air Temperature (°C)** - Environmental condition

### **Smart Features**
- **Input Validation**: Real-time range checking
- **Default Values**: Load typical operating conditions
- **Confidence Scoring**: Prediction reliability assessment
- **Performance Categories**: Excellent/Good/Fair/Poor classification
- **Recommendations**: Actionable suggestions for optimization

## Model Performance

### **Feature Importance Analysis**
Based on comprehensive analysis of 35 features:
- **Random Forest**: 300 trees, optimized hyperparameters
- **Feature Selection**: Physics-based + statistical importance
- **Cross-Validation**: 5-fold CV for robust evaluation

### **Expected Performance**
- **R^2 Score**: 0.85-0.95 (production-ready)
- **RMSE**: <3% of typical NOx conversion values
- **Prediction Time**: <100ms per sample

## File Structure

```
nox-prediction-system/
|
+--- data/
    +--- sample_data.xlsx # Dataset (5,000 samples, 35 features)
+--- requirements.txt        # Python dependencies
+--- README.md              # This file
+--- src/
    +--- train_model.py          # Model training pipeline
    +--- predictor.py            # Prediction engine
    +--- gui.py                  # User-friendly GUI
|
+--- models/                 # Generated after training
    +--- nox_model.pkl      # Trained RandomForest model
    +--- feature_scaler.pkl # Feature scaling parameters
    +--- feature_list.json  # Selected feature names
    +--- feature_importance.csv # Feature importance analysis
```

## Usage Examples

### **Command Line Testing**
```python
from src.predictor import predict_nox_efficiency

# Test case: Highway cruising conditions
test_inputs = {
    'NOx_in_ppm': 350,
    'Engine_Speed': 1800,
    'Eng_torque': 400,
    'Temp_SCR_bed': 380,
    'DEF_inj_rate': 0.12,
    'Exhaust_flow': 140,
    'Temp_DOC_in': 340,
    'Ambient_Air_Tmp': 20
}

result = predict_nox_efficiency(test_inputs)
print(f"NOx CE: {result['nox_ce']}%")
print(f"Category: {result['category']}")
```

### **Batch Prediction**
```python
import pandas as pd
from src.predictor import predict_nox_efficiency

# Load your data
df = pd.read_excel('data/sample_data.xlsx') # Using sample data for demonstration

# Make predictions
predictions = []
for _, row in df.iterrows():
    inputs = row.to_dict()
    result = predict_nox_efficiency(inputs)
    predictions.append(result['nox_ce'])

df['Predicted_NOx_CE'] = predictions
```

## Technical Details

### **Machine Learning Pipeline**
1. **Data Engineering**: Physics-based feature creation
2. **Data Cleaning**: Outlier detection, missing value imputation
3. **Feature Selection**: Combined correlation + importance ranking
4. **Model Training**: RandomForest with optimized hyperparameters
5. **Validation**: Cross-validation + hold-out testing
6. **Deployment**: Scalable prediction API

### **Key Engineered Features**
- **Engine Load Ratio**: Torque-to-speed relationship
- **SCR Temperature Efficiency**: Optimal temperature indicator
- **NOx Loading**: Concentration per unit flow
- **DEF-NOx Ratio**: Dosing efficiency metric

### **Model Architecture**
```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
```

## Performance Optimization

### **For Large Datasets**
- Increase `n_estimators` in `train_model.py`
- Use `n_jobs=-1` for multicore processing
- Consider XGBoost for very large datasets (>100K samples)

### **For Real-Time Prediction**
- Model loads once, predicts in <1ms
- Feature scaling cached for efficiency
- Input validation prevents runtime errors

## Troubleshooting

### **Common Issues**

**"Model artifacts not found"**
```bash
# Solution: Run training first
python src/train_model.py
```

**"Low R^2 performance"**
- Check data quality in `nox_data.xlsx`
- Verify feature ranges are realistic
- Increase `n_estimators` in training

**"GUI input validation errors"**
- Ensure numeric inputs only
- Check values are within specified ranges
- Use "Load Defaults" for valid example

**"ImportError: No module named 'sklearn'"**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

## Data Analysis Results

### **Top Features by Importance**
1. **NOx_in_ppm** (0.0352) - Input concentration
2. **Exhaust_flow** (0.0434) - Flow dynamics
3. **Engine_Speed** (0.0418) - Operating point  
4. **Temp_SCR_bed** (0.0365) - Catalyst activity
5. **Eng_torque** (0.0295) - Load condition

### **Correlation Analysis**
- **Strong Physics-Based Relationships**: Temperature parameters
- **Operating Condition Dependencies**: Engine speed/torque correlation
- **Control Parameter Impact**: DEF injection rate effectiveness

## Production Deployment

### **Integration Options**
1. **Python API**: Direct integration with existing systems
2. **REST Service**: Web-based prediction service
3. **Batch Processing**: Offline analysis of logged data
4. **Real-Time Control**: Integration with engine control units

### **Validation Checklist**
- [ ] R^2 Score > 0.85
- [ ] Cross-validation consistent
- [ ] Feature importance matches domain knowledge
- [ ] Prediction time < 100ms
- [ ] Input validation robust
- [ ] Error handling comprehensive

## Support

For technical support or feature requests:
- Review feature importance analysis in `models/feature_importance.csv`
- Check model performance metrics from training output
- Validate input ranges using GUI tooltips
- Test with default values before custom inputs

---

**Ready to predict NOx conversion efficiency with confidence!**

*Built with Python, scikit-learn, and domain expertise in diesel aftertreatment systems.*
