# SC1015-Mini-Project
## Problem Statement
- To predict phishing links from data available
- How to improve CNN as a predictor

## Dataset
The dataset we used is from Kaggle https://www.kaggle.com/datasets/winson13/dataset-for-link-phishing-detection. This dataset is designed for phishing link detection, containing various features extracted from the URL, the domain, and the  HTML content. 
## Approach
- Feature selection: Correlation, Random Forest Classifier
- Models Used for Machine Learning: Convolutional Neural Network(CNN), CNN with dropout layers
## Folder
### Data Cleaning.ipynb
  The code in this notebook is used to clean the original dataset from Kaggle(dataset_link_phishing.csv). By cleaning abnormal data and duplicate data, we reduce the shape of the dataset from 19431* 87 to 15420*84, gaining a new clean dataset.
### Feature Selection (Random Tree Forest).ipynb
  Because there are still 84 columns (81 features) in the new dataset, we use Random Forest to select the top 10 most relevant features. After dropping the rest of the features, the accuracy of the model has been improved.
### Datavisualisation (selected_dataset ).ipynb

### ML Neural Network.ipynb

We decided to explore something we have not tried before in class to help solve our problem, using a Convolutional Neural Network(CNN).

CNN is a form of machine learning that uses layers of hidden neurons representing weights on certain features to predict a result from an input. In our case, from our feature selection we found the 10 labels most correlated to our result and fed those data into a neural network model with 2 hidden layers of 128 and 256 neurons.

We were able to create a CNN model with a train:test ratio of 0.75:0.25, however when we visualised the training progress of the model against epochs, we found that the accuracy had plateaued after just around 6 epochs. This was a sign of overfitting and prompted us to improve the model.

We decided to then use an additional dropout layer, which randomly drops 20% of the neurons in the 256 layer to reduce dependency on each neuron which will hopefully reduce overfitting.
## Conclusion
- If a link exists in the first 10 pages in google, it is likely not a phishing link
- A suspicious domain surprisingly has little correlation with whether it is phishing or not
- CNN struggles with low number of inputs like 10 labels as it is likely to produce overfitting
- We can quite confidently predict phishing links - 95% accuracy, but false negatives still exist which is likely why link detectors are not widely used as we would still need to manually check for false negatives.
## Contributors
- Data Cleaning, Feature Selection(Random Tree Forest) - Chua Hui Ting Sharon

- ML Neural Network - Huang Bin
## References
- https://www.baeldung.com/cs/ml-relu-dropout-layers
- https://www.datacamp.com/tutorial/cnn-tensorflow-python
- https://www.kaggle.com/code/prashant111/random-forest-classifier-feature-importance
- https://www.youtube.com/watch?v=bDhvCp3_lYw
- https://seaborn.pydata.org/generated/seaborn.violinplot.html
- https://seaborn.pydata.org/generated/seaborn.catplot.html

