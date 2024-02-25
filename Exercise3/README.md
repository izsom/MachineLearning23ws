# Steps to reproduce the results of the second approach

All the images used for training and testing are stored in the data folder. Both folders contain 3 additional folder, one for the images with the original resolution (sharp), one with the blurred images (gaussian_blur) and one for the deblurred images (deblurred). The images with original resolution and shape are stored in the sharp folder.
First, to blur these images the add_gaussian_blur.ipynb needs to be run. This file creates and stores the blurred images inside the inside the gaussian_blur folder.
Second step (optional as the trained models can be found under /outputs) is to train the chosen model running the train.py file or the training_colab.ipynb file from the src folder. This trains the model using the images from the sharp and gaussian_blurred folders.
Third step is to test our model using the test_models.ipynb from the src folder. This testing is based on the images stored in the data/test_data folder. The images that are deblurred in the testing process are saved in the folders under /data/test_data/deblurred. 
The final step is to use the data created during testing for the evaluation, which is done in the evaluation.ipynb (located in the src folder)

Shortly:
1. run add_gaussian_blur.ipynb
2. OPTIONAL! - run train.py or training_colab.ipynb - OPTIONAL!
3. run test_models.ipynb
4. run evaluation.ipynb

# Structure of the project
```
.
├── first_approach                          # Folder containing the first approach
│   ├── models 
│   │   ├── autoencoder_number.keras        # Autoencoder model for number dataset
│   │   └── autoencoder.keras               # Autoencoder model  
│   └── CNN_Autoencoder.ipynb               # Jupyter notebook containing the experiments for the first approach    
├── second_approach                         # Folder containing the second approach
│   ├── data  
│   │   ├── test_data                       # Folder containing the test datasets
│   │   │   ├── deblurred                   # Folder containing the deblurred test datasets
│   │   │   │   ├── chosen_dataset_15x15    # Folder containing the deblurred test images (Blur of 15x15) - Berkeley dataset
│   │   │   │   ├── chosen_dataset_51x51    # Folder containing the deblurred test images (Blur of 51x51) - Berkeley dataset
│   │   │   │   ├── chosen_dataset_simpleae # Folder containing the deblurred test images (Model SimpleAE) - Berkeley dataset
│   │   │   │   └── original_dataset        # Folder containing the list of test images from the original dataset
│   │   │   ├── gaussian_blurred            # Folder containing the gaussian blurred test datasets (Blurred by us)
│   │   │   └── sharp                       # Folder containing the sharp test datasets (original images)
│   │   └── train_data                      # Folder containing the train datasets
│   │       ├── gaussian_blurred            # Folder containing the gaussian blurred train datasets (Blurred by us)
│   │       └── sharp                       # Folder containing the sharp train datasets (original images)
│   ├── outputs  
│   │   ├── loss_chosen_dataset_15x15.png   # Loss plot for the chosen dataset (Blur of 15x15)
│   │   ├── loss_chosen_dataset_51x51.png   # Loss plot for the chosen dataset (Blur of 51x51)
│   │   ├── loss_original_dataset-png       # Loss plot for the original dataset
│   │   ├── model_berkeley_simpleae.pth     # Model trained on the chosen dataset with SimpleAE
│   │   ├── model_chosen_dataset_15x15.pth  # Model trained on the chosen dataset (Blur of 15x15)
│   │   ├── model_chosen_dataset_51x51.pth  # Model trained on the chosen dataset (Blur of 51x51)
│   │   └── model_original_dataset.pth      # Model trained on the original dataset
│   ├── src
│   │   ├── add_gaussian_blur.ipynb         # Notebook containing the code to add gaussian blur to the images
│   │   ├── evaluation.ipynb                # Notebook containing the code for the evaluation of the trained models
│   │   ├── models.py                       # Python file containing the code for the two models (SimpleAE and CNN)
│   │   ├── test_models.ipynb               # Notebook containing the code for deblurring the test images using the trained models
│   │   ├── training_colab.ipynb            # Notebook containing the code for the training of the models on Google Colab
│   │   └── training.py                     # Python file containing the code for the training of the models locally  
│   ├── chosen_img_example.png              # Example of a chosen image from the chosen dataset
│   ├── diff_kernel_chosen_dataset.png      # Plot showing the difference between different blur kernels for the chosen dataset
│   ├── diff_kernel_orig_dataset.jpg        # Plot showing the difference between different blur kernels for the original dataset
│   └── original_img_example.jpg            # Example of a chosen image from the original dataset
└── README.md                               # Readme file     
```

