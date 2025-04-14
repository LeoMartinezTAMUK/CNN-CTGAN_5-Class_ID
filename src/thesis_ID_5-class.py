# NSL-KDD 5-Class Classification using CTGAN and CNN
# Created by: Leo Martinez III from Fall 2024 - Spring 2025
# Created using Spyder v6.0.3 with Python v3.9.21 
#(Note: a separate virtual environment was used for the CTGAN section due to conflicting dependencies)

# Original Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

# RandomUnderSampler Imports
import imblearn
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

# CTGAN Imports (a separate virtual environment was dedicated for GAN to avoid dependency incompatibility)
#from ydata_synthetic.synthesizers.regular import RegularSynthesizer
#from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# Other Imports
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
from keras.models import load_model

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

#%%
"""Load the datasets

Training Dataset
"""

# CHANGE FILE PATHS AS NEEDED
local_path_train = 'data/KDDTrain+.txt'

# --- Importing Train Dataset ---
# NSL-KDD, 43 features, 125973 samples, Multiclass Classification (From text file)
KDDTrain = pd.read_csv(local_path_train, header = None) # CHANGE FILE PATH AS NEEDED
# Column Headings
KDDTrain.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
       'num_access_files', 'num_outbound_cmds', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate', 'class', 'difficulty']

# We will not utilize the 'difficulty' feature for now
KDDTrain.drop('difficulty', axis=1, inplace=True)

"""Testing Dataset"""

# CHANGE FILE PATHS AS NEEDED
local_path_test = 'data/KDDTest+.txt'

# --- Importing Test Dataset ---
# NSL-KDD, 43 features, 22544 samples, Multiclass Classification (From text file)
KDDTest = pd.read_csv(local_path_test, header = None) # CHANGE FILE PATH AS NEEDED
# Column Headings
KDDTest.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
       'num_access_files', 'num_outbound_cmds', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate', 'class', 'difficulty']

# We will not utilize the 'difficulty' feature for now
KDDTest.drop('difficulty', axis=1, inplace=True)

#%%

KDDSynth = pd.read_csv('data/synth_data_5-Class.csv')

KDDSynth['class'].value_counts()

#%%
""" Data Handling """

# We drop 'num_outbound_cmds' from both training and testing dataset because every instance is equal to 0 in both datasets
KDDTrain.drop("num_outbound_cmds",axis=1,inplace=True)
KDDTest.drop("num_outbound_cmds",axis=1,inplace=True)

# We replace all instances with a value of 2 to 1 (1 instead of 0 because of Dr. Mishra's request) because the feature should be a binary value (0 or 1)
KDDTrain['su_attempted'] = KDDTrain['su_attempted'].replace(2, 1)
KDDTest['su_attempted'] = KDDTest['su_attempted'].replace(2, 1)

#%%
# Change training attack labels to their respective attack class for multiclass classification
KDDTrain['class'].replace(['neptune', 'smurf', 'back', 'teardrop', 'pod', 'land'],'DoS',inplace=True) # 6 sub classes of DoS
KDDTrain['class'].replace(['satan', 'ipsweep', 'portsweep', 'nmap'],'Probe',inplace=True) # 4 sub classes of Probe
KDDTrain['class'].replace(['warezclient', 'guess_passwd', 'warezmaster', 'imap', 'ftp_write', 'multihop', 'phf','spy'],'R2L',inplace=True) # 8 sub classes of R2L
KDDTrain['class'].replace(['buffer_overflow', 'rootkit', 'loadmodule','perl'],'U2R',inplace=True) # 4 sub classes of U2R

# Change testing attack labels to their respective attack class for multiclass classification
KDDTest['class'].replace(['neptune', 'apache2', 'processtable', 'smurf', 'back', 'mailbomb', 'pod', 'teardrop', 'land', 'udpstorm'],'DoS',inplace=True) # 10 sub classes of DoS
KDDTest['class'].replace(['mscan', 'satan', 'saint', 'portsweep', 'ipsweep', 'nmap'],'Probe',inplace=True) # 6 sub classes of Probe
KDDTest['class'].replace(['guess_passwd', 'warezmaster', 'snmpguess', 'snmpgetattack', 'httptunnel', 'multihop', 'named', 'sendmail', 'xlock', 'xsnoop', 'ftp_write', 'worm', 'phf', 'imap'],'R2L',inplace=True) # 14 sub classes of R2L
KDDTest['class'].replace(['buffer_overflow', 'ps', 'rootkit', 'xterm', 'loadmodule', 'perl', 'sqlattack'],'U2R',inplace=True) # 7 sub classes of U2R

#%%
"""Data Preprocessing (One Hot Encoding)"""

# Encode class label with LabelEncoder
label_encoder = preprocessing.LabelEncoder()
KDDTrain['class'] = label_encoder.fit_transform(KDDTrain['class'])
KDDTest['class'] = label_encoder.fit_transform(KDDTest['class'])

# Save 'class' columns (append after onehotencoding)
class_train = KDDTrain['class']
class_test = KDDTest['class']

# Drop 'class' column until preprocessing is done
KDDTrain = KDDTrain.drop(['class'], axis=1)
KDDTest = KDDTest.drop(['class'], axis=1)

# Define the columns to OneHotEncode
categorical_columns=['protocol_type', 'service', 'flag']

# Initialize the OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse_output=False)

# Loop through each categorical column and encode it
for column in categorical_columns:
    # Fit and transform the training data then transform the testing data for the current column
    train_encoded = one_hot_encoder.fit_transform(KDDTrain[[column]])
    test_encoded = one_hot_encoder.transform(KDDTest[[column]])

    # Get the categories for the current column
    categories = one_hot_encoder.get_feature_names_out([column])

    # Drop the original categorical column from the original DataFrames
    KDDTrain.drop(columns=[column], inplace=True)
    KDDTest.drop(columns=[column], inplace=True)

    # Add the encoded features to the DataFrames
    KDDTrain[categories] = train_encoded
    KDDTest[categories] = test_encoded

# Define the columns to scale
columns_to_scale=['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

# Scale numerical columns using MinMax
scaler = MinMaxScaler()
for column in columns_to_scale:
    KDDTrain[column] = scaler.fit_transform(KDDTrain[[column]])
    KDDTest[column] = scaler.transform(KDDTest[[column]])

# Add 'class' column back to the dataset as the last column
KDDTrain['class'] = class_train
KDDTest['class'] = class_test

# Used for elaboration on the training data if needed
#print(KDDTrain.shape)
#print(KDDTrain.columns.tolist())

#%%

# (a separate virtual environment was dedicated for GAN to avoid dependency incompatibility)
""" Synthetic Data Creation/Loading """ 
# Synthetic data has already been generated, so training is not needed again

# Defining the training parameters
batch_size = 5000 #5000
epochs = 115 # originally 100
learning_rate = 2e-4 #2e-4
beta_1 = 0.6 #0.5
beta_2 = 0.9 #0.9

#ctgan_args = ModelParameters(batch_size=batch_size, lr=learning_rate, betas=(beta_1, beta_2))

#train_args = TrainParameters(epochs=epochs)

#%%

""" CTGAN (Conditional Tabular Generative Adversarial Network) Synthetic Data Creation """
# Load data for synthetic data creation
data = KDDTrain

num_cols = ['duration','src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations',
       'num_shells','num_access_files', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate']

cat_cols = ['protocol_type', 'service', 'flag', 'class']

#%%
# (a separate virtual environment was dedicated for GAN to avoid dependency incompatibility)
print("Training In Progress...") 

#synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
#synth.fit(data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

print("Training Complete!")
#%%
"""Generate new synthetic data"""
print("Generating Data...")
#synth_data = synth.sample(350000) # originally 300,000
print("Synthetic Data Genererated!")

""" Optional to Save Synth Dataset"""

# Use this code if you want to save the synthetic data for future use or examination
#synth_data.to_csv('synth_data_5-Class.csv', index=False)

#%%

# This code does not need a specific virtual environment
""" Optional to Import Synthetic Data"""

# Use this code if you already have already generated synthetic data
synth_data = pd.read_csv('data/synth_data_5-Class.csv') # Change path as needed

#%%
"""Drop Unneeded Synthetic Samples"""

# There is already a large sample size for class '4' and '0', no need for additional synthetic data
# Drop rows with class '4' or '0'
KDDSynth = synth_data[(synth_data['class'] != 4) & (synth_data['class'] != 0)]

# Distribution of classes in dataset after synthetic concatenation
print(KDDSynth['class'].value_counts())

#%%

# NOTE: Converting Data to Images can be very computationally expensive and/or time consuming

""" Convert Training Data to Images """

def train_sample_to_image(index, dataset, image_size=(11, 11)):
    # Extract a single sample
    sample = dataset.iloc[index].drop('class').values

    # Reshape the sample to match the image size
    image = sample.reshape(image_size)

    # Plot the image
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Hide axes

    # Construct the file path using the index
    file_path = f'images/train/{index}.png'

    # Save the image as a .png file
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)  # Save image without extra whitespace
    plt.close()  # Close the plot to avoid display

# Example usage
#train_sample_to_image(0, KDDTrain)

# Loop through all testing samples and generate images
for idx in range(len(KDDTrain)):
    train_sample_to_image(idx, KDDTrain)

#%%

""" Convert Testing Data to Images """

def test_sample_to_image(index, dataset, image_size=(11, 11)):
    # Extract a single sample
    sample = dataset.iloc[index].drop('class').values

    # Reshape the sample to match the image size
    image = sample.reshape(image_size)

    # Plot the image
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Hide axes

    # Construct the file path using the index
    file_path = f'images/test/{index}.png'

    # Save the image as a .png file
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)  # Save image without extra whitespace
    plt.close()  # Close the plot to avoid display

# Example usage
#test_sample_to_image(0, KDDTest)

# Loop through all testing samples and generate images
for idx in range(len(KDDTest)):
    test_sample_to_image(idx, KDDTest)
    
#%%

""" Convert Synthetic Data (For Training) to Images """

def synth_sample_to_image(index, dataset, image_size=(11, 11)):
    # Extract a single sample
    sample = dataset.iloc[index].drop('class').values

    # Reshape the sample to match the image size
    image = sample.reshape(image_size)

    # Plot the image
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Hide axes

    # Construct the file path using the index
    file_path = f'images/synth/synth_{index}.png'

    # Save the image as a .png file
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)  # Save image without extra whitespace
    plt.close()  # Close the plot to avoid display

# Example usage
#test_sample_to_image(0, KDDSynth)

# Loop through all testing samples and generate images
for idx in range(len(KDDSynth)):
    synth_sample_to_image(idx, KDDSynth)
    
#%%

""" Generate Proper Labels for Corresponding Images (Synthetic Samples Only)"""

# Function to create label CSV
def synth_create_label_csv(dataset, output_file):
    # Create a DataFrame for labels
    labels_df = dataset[['class']].copy()
    labels_df.reset_index(drop=True, inplace=True) # Reset index without keeping the old index
    labels_df.reset_index(inplace=True)
    labels_df['filename'] = labels_df['index'].apply(lambda x: f'synth_{x}.png')
    labels_df = labels_df[['filename', 'class']]
    
    # Save to CSV
    labels_df.to_csv(output_file, index=False)

# Corresponding dataset class to synth images
synth_create_label_csv(KDDSynth, 'data/synth_labels.csv')
    
#%%

""" Generate Proper Labels for Corresponding Images (Real Samples Only (Test and Train))"""

# Function to create label CSV
def create_label_csv(dataset, output_file):
    # Create a DataFrame for labels
    labels_df = dataset[['class']].copy()
    labels_df.reset_index(inplace=True)
    labels_df['filename'] = labels_df['index'].apply(lambda x: f'{x}.png')
    labels_df = labels_df[['filename', 'class']]
    
    # Save to CSV
    labels_df.to_csv(output_file, index=False)

# Corresponding dataset class to train images
create_label_csv(KDDTrain, 'data/train_labels.csv')

# Corresponding dataset class to test images
create_label_csv(KDDTest, 'data/test_labels.csv')

#%%

# Note: Loading the images can be time consuming

""" Prepare Image Data for CNN """

from PIL import Image
import os

# Load labels
train_labels_df = pd.read_csv('data/train_labels.csv')
test_labels_df = pd.read_csv('data/test_labels.csv')
synth_labels_df = pd.read_csv('data/synth_labels.csv')

def load_images_and_labels(image_folder, labels_df, image_size=(11, 11)):
    images = []
    labels = []
    for _, row in labels_df.iterrows():
        image_path = os.path.join(image_folder, row['filename'])
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = image.resize(image_size)  # Resize to desired size
            image = np.array(image) / 255.0  # Normalize
            images.append(image)
            labels.append(row['class'])
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    
    # Convert list to numpy array before reshaping
    images = np.array(images)
    # Reshape images to include the channel dimension (1 for grayscale)
    images = images.reshape(-1, image_size[0], image_size[1], 1)
    
    return images, np.array(labels)

# Load training and testing images
X_train, y_train = load_images_and_labels('images/train', train_labels_df)
X_test, y_test = load_images_and_labels('images/test', test_labels_df)
X_synth, y_synth = load_images_and_labels('images/synth', synth_labels_df)

#%%

# Shuffle the data to enhance generalizability
from sklearn.utils import shuffle

# Combine training data with synthetic data
X_train_combined = np.concatenate((X_train, X_synth), axis=0)
y_train_combined = np.concatenate((y_train, y_synth), axis=0)

X_train_combined, y_train_combined = shuffle(X_train_combined, y_train_combined, random_state=0)

#%%

""" Undersampling (Synthetic Data Included) """

# Random Under Sampling to even out sample sizes amongst classes
# 0 = DoS, 1 = Probe, 2 = R2L, 3 = U2R, 4 = normal | lexicographic order
under_sampling_strategy = {4: 19000, 0: 30500, 1: 29500}

# Reshape X_train for undersampling
num_samples, height, width, channels = X_train_combined.shape
X_train_reshaped = X_train_combined.reshape(num_samples, -1)

# Perform undersampling
under = RandomUnderSampler(sampling_strategy=under_sampling_strategy, random_state=42) 
X_train_resampled, y_train_resampled = under.fit_resample(X_train_reshaped, y_train_combined)

# Reshape X_train back to 4D (samples, height, width, channels) after undersampling
X_train_resampled = X_train_resampled.reshape(-1, height, width, channels)

# Update X_train and y_train with resampled data
X_train = X_train_resampled
y_train = y_train_resampled

#%%

""" Loading the Saved CNN Model """

#del model
    
# Load the model from file
model = load_model("cnn_model_best_0.8699876070022583.h5")

# Summary of the model
model.summary()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Make predictions
y_pred = model.predict(X_test)

# Convert predictions from probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Convert true labels to integers
y_test_labels = y_test

# Evaluate the overall performanc
print(classification_report(y_test,y_pred_labels, digits=4))
MCC = matthews_corrcoef(y_test, y_pred_labels)
print("MCC: ", MCC)

# Generate normalized confusion matrix
matrix = confusion_matrix(y_test_labels, y_pred_labels, normalize=None)

# Plot confusion matrix as heatmap
plt.figure(figsize=(5, 5))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['DoS', 'Probe', 'R2L', 'U2R', 'normal'], 
            yticklabels=['DoS', 'Probe', 'R2L', 'U2R', 'normal'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.savefig('confusion_matrix_heatmap.png', dpi=400, bbox_inches='tight')
plt.show()

# Generate normalized confusion matrix
normalized_matrix = confusion_matrix(y_test_labels, y_pred_labels, normalize='true')

# Plot normalized confusion matrix as heatmap
plt.figure(figsize=(5, 5))
sns.heatmap(normalized_matrix, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=['DoS', 'Probe', 'R2L', 'U2R', 'normal'], 
            yticklabels=['DoS', 'Probe', 'R2L', 'U2R', 'normal'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix Heatmap')
plt.savefig('normalized_confusion_matrix_heatmap.png', dpi=400, bbox_inches='tight')
plt.show()


#%%

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Convert true labels to one-hot encoding (it was originally in labelencoded)
y_test_onehot = label_binarize(y_test, classes=[0, 1, 2, 3, 4])

# Compute ROC curve and AUC for each class
fpr = {}  # False positive rate
tpr = {}  # True positive rate
roc_auc = {}  # AUC score

for i in range(5):  # 5 classes
    fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curves
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green', 'orange', 'purple']
class_labels = ['DoS', 'Probe', 'R2L', 'U2R', 'normal']

for i, color in zip(range(5), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {class_labels[i]} (AUC = {roc_auc[i]:.2f})')

# Plot the random classifier line (diagonal)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Configure the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)

# Save and show the plot
plt.savefig('multi_class_auc.png', dpi=400, bbox_inches='tight')
plt.show()

#%%

#from sklearn.model_selection import KFold
#from sklearn.metrics import accuracy_score, classification_report
#import numpy as np
#from tensorflow.keras.models import load_model

""" Perform 10-Fold Cross-Validation """

# Load pre-trained model
model = load_model("cnn_model_best_0.8699876070022583.h5")

# Initialize K-Fold Cross-Validation with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=5)

# Store accuracies for each fold
accuracies = []

# Perform 10-Fold Cross-Validation
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Evaluate model on the validation fold
    y_pred_val = model.predict(X_val_fold)
    y_pred_labels = np.argmax(y_pred_val, axis=1)  # Convert probabilities to class labels

    # Calculate accuracy for the current fold
    accuracy = accuracy_score(y_val_fold, y_pred_labels)
    accuracies.append(accuracy)

    # Optionally print metrics for each fold
    print(f"Fold accuracy: {accuracy:.4f}")
    print(classification_report(y_val_fold, y_pred_labels, digits=4))

# Calculate and print the mean accuracy across all folds
mean_accuracy = np.mean(accuracies)
print(f"Mean 10-Fold CV Accuracy: {mean_accuracy:.4f}")

# Save the results to a file
with open('cv_results.txt', 'a') as outputFile:
    outputFile.write(f'10-CV = {mean_accuracy:.4f}\n')

#%%

""" Convolutional Neural Network Implementation for 5-class classification """
    
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
    
# Define input shape for the images (11x11 pixels, grayscale (1 channel))
input_shape = (11, 11, 1)

# Input layer
inputs = Input(shape=input_shape)

# Convolutional layers
conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu')(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
dropout1 = Dropout(0.25)(pool1)  # Dropout to prevent overfitting

conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(dropout1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

# Flatten layer
flatten = Flatten()(pool2)

# Fully connected layers
dense1 = Dense(256, activation='relu')(flatten)
outputs = Dense(5, activation='softmax')(dense1)  # Output layer with 5 units for 5 classes

# Define the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model with accuracy tracking
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model and store history
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    validation_split=0.20, 
                    batch_size=256, 
                    verbose=1)

# Print accuracy values from training (optional)
#for epoch, (train_acc, val_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy']), 1):
#    print(f"Epoch {epoch}: Training Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}")

# Plot Training and Validation Accuracy
plt.figure(figsize=(8, 6), dpi=400)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png', dpi=400)  # Save the figure with 400 DPI
plt.show()

# Save the model after training
#model.save("cnn_model.h5")

#----------------------------------------------------------------------------

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Make predictions
y_pred = model.predict(X_test)

# Convert predictions from probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Convert true labels to integers
y_test_labels = y_test

# Generate confusion matrix
matrix = confusion_matrix(y_test_labels, y_pred_labels)

# Plot confusion matrix as heatmap
plt.figure(figsize=(5, 5))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['DoS', 'Probe', 'R2L', 'U2R', 'normal'], 
            yticklabels=['DoS', 'Probe', 'R2L', 'U2R', 'normal'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

#----------------------------------------------------------------------------

# Evaluate the overall performance
print(classification_report(y_test,y_pred_labels, digits=4))
