# Wine Quality Classification using AWS SageMaker

## 📋 Project Overview

This project implements a machine learning pipeline for classifying wine quality using AWS SageMaker. The implementation compares two different algorithms - **XGBoost** and **Linear Learner** - to predict wine quality categories based on various wine characteristics.

## 🎯 Objectives

- Build and deploy machine learning models on AWS SageMaker
- Classify wine quality into three categories: Low, Average, and High
- Compare performance between XGBoost and Linear Learner algorithms
- Implement end-to-end ML pipeline including data preprocessing, training, deployment, and evaluation

## 📊 Dataset

**Dataset**: White Wine Quality Dataset
- **Source**: Stored in S3 bucket (`s3://p4-hao/winequality-white.csv`)
- **Features**: Various wine characteristics (chemical properties)
- **Target Variable**: Wine quality scores converted to categorical labels:
  - **Low Quality** (0-4): Label 2
  - **Average Quality** (5-7): Label 0  
  - **High Quality** (8-10): Label 1

## 🏗️ Architecture

### Data Pipeline
1. **Data Loading**: Read wine quality dataset from S3
2. **Data Preprocessing**: 
   - Convert quality scores to categorical labels
   - Label encoding for machine learning
3. **Data Splitting**: 
   - Training set (80%)
   - Validation set (10%)
   - Test set (10%)
4. **Data Storage**: Upload processed datasets back to S3

### Model Training & Deployment
- **Platform**: AWS SageMaker
- **Instance Type**: ml.m4.xlarge
- **Algorithms**: XGBoost and Linear Learner
- **Deployment**: Real-time endpoints for inference

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas scikit-learn numpy boto3 sagemaker
```

### AWS Setup Requirements

1. **AWS Account** with SageMaker access
2. **IAM Role** with SageMaker execution permissions
3. **S3 Bucket** for data storage (configured as `p4-hao`)
4. **SageMaker Notebook Instance** or local environment with AWS credentials

### Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wine-quality-sagemaker.git
cd wine-quality-sagemaker
```

2. Configure AWS credentials:
```bash
aws configure
```

3. Update the S3 bucket name in the code if needed:
```python
bucket_name = 'your-bucket-name'
```

4. Run the main script:
```python
python main.py
```

## 📈 Model Specifications

### XGBoost Model
```python
Hyperparameters:
- objective: 'multi:softmax'
- num_class: 3
- num_round: 100
```

### Linear Learner Model
```python
Hyperparameters:
- predictor_type: 'multiclass_classifier'
- num_classes: 3
```

## 📊 Evaluation Metrics

The models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Macro-averaged precision across all classes
- **Recall**: Macro-averaged recall across all classes

## 📁 Output Files

The pipeline generates the following files:

1. **Dataset Files**:
   - `train.csv`: Training dataset
   - `validation.csv`: Validation dataset
   - `test.csv`: Test dataset

2. **Classification Results**:
   - `classification results_XGBoost model.csv`: XGBoost predictions on test set
   - `classification results_Linear Learner model.csv`: Linear Learner predictions on test set

3. **Performance Metrics**:
   - `performance_metrics.csv`: Comparative performance metrics for both models

All files are automatically uploaded to the configured S3 bucket.

## 🔧 Code Structure

```
main.py
├── Data Loading & Preprocessing
├── Data Splitting & S3 Upload
├── XGBoost Model Training & Deployment
│   ├── Model Configuration
│   ├── Training Job Execution
│   ├── Endpoint Deployment
│   ├── Prediction & Evaluation
│   └── Endpoint Cleanup
├── Linear Learner Model Training & Deployment
│   ├── Model Configuration
│   ├── Training Job Execution
│   ├── Endpoint Deployment
│   ├── Prediction & Evaluation
│   └── Endpoint Cleanup
└── Performance Comparison & Results Export
```

## ⚡ Key Features

- **Automated Pipeline**: Complete ML pipeline from data preprocessing to model evaluation
- **Cloud-Native**: Fully implemented on AWS SageMaker infrastructure
- **Model Comparison**: Side-by-side comparison of XGBoost vs Linear Learner
- **Resource Management**: Automatic endpoint cleanup to prevent unnecessary costs
- **Comprehensive Evaluation**: Multiple metrics for thorough model assessment
- **Result Export**: Structured output files for further analysis

## 📋 Usage Example

```python
# The main script automatically:
# 1. Loads and preprocesses data
# 2. Trains both models
# 3. Deploys endpoints
# 4. Makes predictions
# 5. Evaluates performance
# 6. Saves results to S3
# 7. Cleans up resources
```

## 💰 Cost Optimization

- **Endpoint Management**: Endpoints are automatically deleted after use to minimize costs
- **Instance Selection**: Uses cost-effective ml.m4.xlarge instances
- **Resource Cleanup**: Proper cleanup of all AWS resources

## 🛠️ Troubleshooting

### Common Issues:

1. **S3 Permissions**: Ensure your IAM role has S3 read/write access
2. **SageMaker Permissions**: Verify SageMaker execution role permissions
3. **Instance Availability**: ml.m4.xlarge instances should be available in your region
4. **Bucket Names**: Update bucket names to match your AWS setup

### Error Handling:

The code includes error handling for:
- S3 file upload operations
- Model training job failures
- Endpoint deployment issues

## 📊 Expected Results

Both models will output:
- Training and validation metrics
- Test set predictions
- Performance comparison metrics
- Detailed classification results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is available for educational and research purposes.

## 📞 Contact

For questions or issues, please open an issue in this repository.

---

*This project demonstrates cloud-based machine learning using AWS SageMaker for wine quality classification, comparing different algorithms and providing comprehensive evaluation metrics.*
