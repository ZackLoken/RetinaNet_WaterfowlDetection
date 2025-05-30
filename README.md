# RetinaNet_WaterfowlDetection
Repository with PyTorch code for training RetinaNet w/ ResNet-50 backbone to identify waterfowl in drone images and videos. Users can clone this repository, load the corresponding Conda environment from environment.yml, download the image and annotation data (link below), and then change the folder paths in RetinaNet_ResNet50_PyTorch_CustomDataset.ipynb or RetinaNet_ResNet50_PyTorch_CustomDataset.py to the directory where the data was downloaded. If using a GPU, CUDA Toolkit version 12.4 must be downloaded following the instructions here: (https://developer.nvidia.com/cuda-12-4-0-download-archive).

![alt text](https://github.com/ZackLoken/RetinaNet_WaterfowlDetection/blob/main/TestResults.png)  

Sample predictions for four images in the test dataset. Each image pair shows the ground truth bounding boxes (left image) and the predicted bounding boxes (right image). For each detection, the model predicts the class, location, and confidence score. Predictions have been post-processed using soft non-maximum suppression and filtered to remove any predictions with confidence scores below 0.5. The model correctly identified two NSHO and one Hen (A); one GWTE, one MALL, two NOPI, and two Hen (B); one REDH, 12 RNDU, and 3 Hen (C); and five AMCO and one GADW (D).

Repository Contents:
 
 * RetinaNet_ResNet50_PyTorch_CustomDataset.ipynb -- Jupyter notebook containing code for performing PyTorch object detection on a custom dataset. Specifically, this notebook contains code for pre-processing image and annotation data, tuning hyperparameters using Bayesian Optimization, gradient accumulation enabled fine-tuning of RetinaNet and ResNet-50 pre-trained on COCO and ImageNet, respectively, final model inference on the test dataset, and visualizing model predictions on test set images. Also includes functions for plotting training and validation metrics, confusion matrices for the test dataset, and running inference on videos using the fine-tuned model.
 
 * RetinaNet_ResNet50_PyTorch_CustomDataset.py -- Python script for running Jupyter Notebook from the command line. Retains all necessary code to run the analysis, minus the plotting code. 
 
 * coco_eval.py -- COCO style dataset evaluation tools
 
 * coco_utils.py -- COCO style dataset utilities
 
 * engine_gradientAccumulation.py -- Gradient accumulation enabled PyTorch object detection model training and evaluation engines
 
 * environment.yml -- YAML for cloning the Conda environment for this repo
 
 * transforms.py -- PyTorch object detection transformation functions
 
 * utils.py -- PyTorch utils for object detection training and evaluation engines

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
