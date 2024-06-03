# This program is verified to run in Notebook instance on AWS Sagemaker

import io
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import boto3

import sagemaker
from sagemaker import get_execution_role

data = pd.read_csv('s3://p4-hao/winequality-white.csv', sep=';')

data['quality'] = pd.cut(data['quality'], bins=[0, 4, 7, 10], labels=['Low', 'Average', 'High'])

# Encode 'Average' as 0, 'High' as 1, 'Low' as 2
label_encoder = sklearn.preprocessing.LabelEncoder()
data['quality_encoded'] = label_encoder.fit_transform(data['quality'])

# Split the dataset into features and target ndarray
y = data['quality_encoded'].values
X = data.drop(['quality', 'quality_encoded'], axis=1).values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Save the train, validation, and test sets to CSV files
pd.concat([pd.DataFrame(y_train), pd.DataFrame(X_train)], axis=1).to_csv('train.csv', index=False, header=False)
pd.concat([pd.DataFrame(y_test), pd.DataFrame(X_test)], axis=1).to_csv('test.csv', index=False, header=False)
pd.concat([pd.DataFrame(y_val), pd.DataFrame(X_val)], axis=1).to_csv('validation.csv', index=False, header=False)

# Initialize a boto3 client for S3
s3_client = boto3.client('s3')

# Specify your S3 bucket and the folder path you want to use
bucket_name = 'p4-hao'
folder_path = 'csv_files/'


# Function to upload files to a specified bucket
def upload_file_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        print(f"File {file_name} uploaded to {bucket}/{object_name}")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")


# Upload the training data CSV file to S3
upload_file_to_s3('train.csv', bucket_name, folder_path + 'train.csv')

# Upload the validation data CSV file to S3
upload_file_to_s3('validation.csv', bucket_name, folder_path + 'validation.csv')

# upload the test data CSV file
upload_file_to_s3('test.csv', bucket_name, folder_path + 'test.csv')

sagemaker_session = sagemaker.Session()
role = get_execution_role()


# --- XGBoost model training and deployment ---
# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
xgboost_image = sagemaker.image_uris.retrieve(framework='xgboost', region=sagemaker_session.boto_region_name,
                                              version='1.7-1')

# construct a SageMaker estimator that calls the xgboost-container
xgb_estimator = sagemaker.estimator.Estimator(
    image_uri=xgboost_image,
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=f's3://{bucket_name}/output',
    sagemaker_session=sagemaker_session,
    hyperparameters={
        'objective': 'multi:softmax',
        'num_class': 3,
        'num_round': 100
    })

# define paths to the training and validation datasets
train_input_xgb = sagemaker.inputs.TrainingInput(s3_data=f's3://{bucket_name}/{folder_path}train.csv',
                                                 content_type='csv')
validation_input_xgb = sagemaker.inputs.TrainingInput(s3_data=f's3://{bucket_name}/{folder_path}validation.csv',
                                                      content_type='csv')

# execute the XGBoost training job
xgb_estimator.fit({'train': train_input_xgb, 'validation': validation_input_xgb})

# Deploy the trained model to an Amazon SageMaker endpoint.
xgb_predictor = xgb_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge',
    serializer=sagemaker.serializers.CSVSerializer(),
    deserializer=sagemaker.deserializers.CSVDeserializer()
)

# convert X_test into a CSV format string without actually writing to disk
test_csv = io.StringIO()

pd.DataFrame(X_test).to_csv(test_csv, header=False, index=False)
xgb_predictions = xgb_predictor.predict(test_csv.getvalue())

# convert xgb_predictions from 2D list to 1D ndarray
xgb_predictions_converted = np.array([int(float(pred[0])) for pred in xgb_predictions])

print('XGBoost performance metrics:')
accuracy_xgb = accuracy_score(y_test, xgb_predictions_converted)
precision_xgb = precision_score(y_test, xgb_predictions_converted, average='macro')
recall_xgb = recall_score(y_test, xgb_predictions_converted, average='macro')

print(f"Accuracy of XGBoost model: {accuracy_xgb:.2f}")
print(f"Precision of XGBoost model: {precision_xgb:.2f}")
print(f"Recall of XGBoost model: {recall_xgb:.2f}")

# Delete the XGBoost endpoint after use
xgb_predictor.delete_endpoint()
print("XGBoost endpoint deleted.")

# create classification result
feature_columns = data.columns.drop(['quality', 'quality_encoded'])  # Save feature names before dropping

X_train, X_temp, y_train, y_temp = train_test_split(data[feature_columns].values, y, test_size=0.2, random_state=42,
                                                    stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

X_test_df = pd.DataFrame(X_test, columns=feature_columns)

label_map = {0: 'Average Quality', 1: 'High Quality', 2: 'Low Quality'}
xgb_predicted_labels = pd.Series(xgb_predictions_converted).map(label_map)

xgb_result_df = X_test_df.copy()

xgb_result_df['Predicted Quality'] = xgb_predicted_labels

xgb_result_df.to_csv('classification results_XGBoost model.csv')

upload_file_to_s3('classification results_XGBoost model.csv', bucket_name,
                  'p4-Hao/classification results_XGBoost model.csv')


# --Linear Learner model training and deployment--
linear_learner_image = sagemaker.image_uris.retrieve(framework='linear-learner',
                                                     region=sagemaker_session.boto_region_name)

linear_learner = sagemaker.estimator.Estimator(image_uri=linear_learner_image,
                                               role=role,
                                               instance_count=1,
                                               instance_type='ml.m4.xlarge',
                                               output_path=f's3://{bucket_name}/output',
                                               sagemaker_session=sagemaker_session,
                                               hyperparameters={
                                                   'predictor_type': 'multiclass_classifier',
                                                   'num_classes': 3
                                               }
                                               )

train_input_linear = sagemaker.inputs.TrainingInput(s3_data=f's3://{bucket_name}/{folder_path}train.csv',
                                                    content_type='text/csv')
validation_input_linear = sagemaker.inputs.TrainingInput(s3_data=f's3://{bucket_name}/{folder_path}validation.csv',
                                                         content_type='text/csv')

linear_learner.fit({'train': train_input_linear, 'validation': validation_input_linear})

linear_predictor = linear_learner.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge',
    serializer=sagemaker.serializers.CSVSerializer(),
    deserializer=sagemaker.deserializers.CSVDeserializer()
)

# convert X_test into a CSV format string without actually writing to disk
test_csv = io.StringIO()
pd.DataFrame(X_test).to_csv(test_csv, header=False, index=False)
linear_predictions = linear_predictor.predict(test_csv.getvalue())

# convert linear_predictions from 2D list to 1D ndarray
linear_predictions_converted = np.array([int(float(pred[0])) for pred in linear_predictions])

print('Linear learner performance metrics:')
accuracy_linear = accuracy_score(y_test, linear_predictions_converted)
precision_linear = precision_score(y_test, linear_predictions_converted, average='macro')
recall_linear = recall_score(y_test, linear_predictions_converted, average='macro')

print(f"Accuracy of XGBoost model: {accuracy_linear:.2f}")
print(f"Precision of XGBoost model: {precision_linear:.2f}")
print(f"Recall of XGBoost model: {recall_linear:.2f}")

# Delete the Linear Learner endpoint after use
linear_predictor.delete_endpoint()
print("Linear Learner endpoint deleted.")

# create classification result
linear_predicted_labels = pd.Series(linear_predictions_converted).map(label_map)

linear_result_df = X_test_df.copy()

linear_result_df['Predicted Quality'] = linear_predicted_labels

linear_result_df.to_csv('classification results_Linear Learner model.csv')

upload_file_to_s3('classification results_Linear Learner model.csv', bucket_name,
                  'p4-Hao/classification results_Linear Learner model.csv')


# create performance metrices
performance_metrics_df = pd.DataFrame({'Model': ['XGBoost', 'Linear Learner'],
                                       'Accuracy': [accuracy_xgb, accuracy_linear],
                                       'Precision': [precision_xgb, precision_linear],
                                       'Recall': [recall_xgb, recall_linear]})

performance_metrics_df.to_csv('performance_metrics.csv', index=False)

upload_file_to_s3('performance_metrics.csv', bucket_name, 'p4-Hao/performance_metrics.csv')
