
# Structure of the project
```
.
├── preprocessed-datasets             # Folder containing the datasets that have been preprocessed in Exercise 1 
│   ├── bank_marketing_prepro.csv   
│   ├── CongressionVoting_prepro.csv         
│   └── wine_quality_prepro.csv    
├── results                           # Folder containing the results of the experiments
│   ├── cv_grid_search_results.csv    # Results of the grid search for our own NN implementation
│   ├── cv_local_search_results.csv   # Results of the local search for our own NN implementation
│   ├── MLP.csv                       # Results of the MLP classifier
│   └── traditional_results.csv       # Results of the traditional classifier -> RandomForestClassifier
├── results_plots                     # Folder containing plots generated in comparison.ipynb 
├── comparison.ipynb                  # Notebook containing the comparison of the classifiers
├── mlp.ipynb                         # Notebook containing the MLP classifier
├── nn_grid_search.ipynb              # Notebook containing the grid search for our own NN implementation
├── nn_implementation.py              # Our own NN implementation with torch
├── nn_local_search.ipynb             # Notebook containing the local search for our own NN implementation
├── rf_class.ipynb                    # Notebook containing the training and prediction for RandomForestClassifier (based on traditional_class.py)                            
└── traditional_class.py              # Notebook containing the functions for the RandomForestClassifier pipeline
```

