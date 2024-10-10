# CNN-Matlab-code
Overview

This repository contains three primary MATLAB scripts for processing, labeling, feature extraction, training, and evaluating a dataset of AFM images. The workflow includes image augmentation, labeling, and training classifiers on features extracted from the dataset. This README will walk you through the purpose and usage of each script.

Scripts Overview

Data Preparation and Augmentation 

File Name: dataset_preparation.m

Description: This script is responsible for preparing the dataset by augmenting images and creating regions of interest (ROIs) for later analysis. It includes image subdivision, augmentation (rotation and reflection), and saving ROIs into structured folders for different classes. The script outputs both training and testing datasets categorized into distinct classes.

Labeling and Dataset Validation 

File Name: image_labeling.m

Description: This script loads the image data into the MATLAB Image Labeler app to manually label the ROIs. The user can add different labels and attributes for further analysis and model training. It allows easy access to annotated datasets to improve the model's feature extraction and classification accuracy.

Feature Extraction, Training, and Evaluating Classifiers 

File Name: feature_extraction_classification.m

Description: This script extracts features from the labeled images and trains different classifiers to evaluate their performance. It allows the user to set parameters such as neighborhood pixel radius and augmentation options. The classifiers are trained on features extracted using the SSN (Spatial Scattering Network) approach. Results of classifier accuracy and feature selection are stored for each class.

Dependencies and Setup

MATLAB Version: The scripts are compatible with MATLAB R2020b or later.

Required Toolboxes: Image Processing Toolbox, Parallel Computing Toolbox.

Dataset Path: Update the dataset paths in each script to the location on your system. Examples are provided in the scripts, but ensure to replace with your actual paths.

How to Use

Data Preparation:

Run the first script (dataset_preparation.m) to prepare your dataset. Adjust the parameters if necessary for augmentation, number of rotations, and ROIs per image.

Make sure to update the paths for source images, and output folders to store labeled images.

Image Labeling:

Use image_labeling.m to load the images in the Image Labeler app. Manually go through each image and annotate the regions of interest based on the attributes defined.

Save the labels and attributes after labeling.

Feature Extraction and Classifier Training:

Run feature_extraction_classification.m for extracting features from the annotated images.

Use the SSN-based feature extraction and classifier models (e.g., LDA) to train and test the model on the dataset.

Accuracy metrics for each classifier are saved for further evaluation. Experiment with different neighborhood radius (rn) combinations to find the best-performing configuration.

Project Structure

Raw source images: Contains the raw dataset images for processing and labeling.

LabelledImagesROIsDataBase: Contains labeled datasets with ROIs subdivided.

Results: Stores output results, classifier performance metrics, and best-performing configurations.

Training Neural Network: Directory for training scripts and saved models.

Notes and Best Practices

Make sure to replace all paths (C:\Users\nader\...) with correct paths on your local system.

Before running the feature extraction, confirm that labeled data is saved in the proper format and location.

Each classifier is trained using different features extracted from the dataset. To improve accuracy, experiment with augmentation methods and classifier hyperparameters.

Future Enhancements

Cross Validation: Implement k-fold cross-validation to improve classifier reliability.

Automation: Automate the labeling and ROI extraction for improved workflow.

New Classifiers: Incorporate additional classifier models like SVM or Neural Networks to evaluate and compare performance.

License

This project is licensed under the MIT License.

Contact

If you have any questions or issues with the scripts, please feel free to reach out to the repository maintainer at [your-email@example.com].

