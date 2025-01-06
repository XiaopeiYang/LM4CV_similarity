This is the official implementation of our ICCV paper **Learning Concise and Descriptive Attributes for Visual Recognition.**

## Requirements
+ torch == 2.0.1
+ python 3.9.13
+ torchvision == 0.15.2

## Attributes queired for each class
We put the attributes quried for fungi class with GPT3 in the file `/data/fungi/fungi_attributes.txt`.

## Training/testing samples split for each class
The split depends on the file `/data/fungi/split_zhou_Fungi.json` (100 samples per class for training and 100 samples per class for testing).

## Parameters
These key parameters are in the file `/configs/fungi.yaml` 

The following key parameters are available for customization:

- **cluster_feature_method**: Choose one from [kmeans, random, linear]. "Linear" refers to our method.
- **model_size**: Set the size of the CLIP model.
- **mahalanobis**: Enable or disable Mahalanobis distance regularization.
- **division_power**: Control the strength of Mahalanobis constraints.
- **reinit**: Decide whether to initialize the model with weights from image training features.
- **num_attributes**: Specify the number of attributes selected for classification.

Please make sure to adjust these parameters according to your requirements.

The following are the key parameters for different methods of processing images:
- **use_patches**: Choose False to keep the full image; choose True to divide the image into crops.
- **IMP_SAMP_fix**: Choose False to create crops with the same fixed sizes; choose True to create crops with randomly different sizes.

The following are the key parameters for different testing:

Testing for classification accuracy:
- **base_ratio: 1.0**: Choosing 1.0 here allows training on all classes.
- **use_few_shot**: When choosing True, it determines whether to use few-shot learning for 10-shot.

  run ```python main.py  ```

Testing image similarity (R@1) based on attribute space:
- **evaluate_with_novel_classes** : choosing False means testing on whole classes, choosing True means testing only on novel classes.
- **base_ratio**: Choose one from [0.1 ~ 0.9]. Choosing 0.5 means 3 base classes for training; choosing 0.7 means 4 base classes for training; choosing 0.4 means 2 base classes for training.
- **with_attributes**: Choosing True means evaluation based on attribute space.
- **umap**: Choosing True means generating UMAP images based on image similarity.

  run```python test.py  ```

Testing the similarity of each attribute to the class:

- **attribute_similarity_scores** : choosing True means calculating the similarity of each attribute to the class
- **evaluate_with_novel_classes: False**: choosing False to test on whole classes

  run ```python test.py  ```

  Generate and save histogram images to `./histogram/random` and `./histogram/all`:

  `./histogram/random`: Randomly select one image from each class and calculate the similarity score with each attribute.

  `./histogram/all`: Select all images from each class and calculate the similarity score with each attribute.

Testing with knn-classifier:

- **nn-classifier** : choosing True means testing with knn-classifier 
-  **base_ratio**: Choose one from [0.1 ~ 0.9]. Choosing 0.5 means 3 base classes for training; choosing 0.7 means 4 base classes for training; choosing 0.4 means 2 base classes for training.

  run ```python knn.py  ```




