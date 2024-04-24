# SC1015-Mini-Project
## Problem Statement
To predict whether we can detect phishing in the links
Which model is the best to predict phishing

## Dataset
The dataset we used is from Kaggle https://www.kaggle.com/datasets/winson13/dataset-for-link-phishing-detection. This dataset is designed for phishing link detection, containing various features extracted from the URL, the domain, and the  HTML content. 
## Approach
## Folder
- ML Neural Network.ipynb

We decided to explore something we have not tried before in class to help solve our problem, using a Convolutional Neural Network(CNN).

CNN is a form of machine learning that uses layers of hidden neurons representing weights on certain features to predict a result from an input. In our case, from our feature selection we found the 10 labels most correlated to our result and fed those data into a neural network model with 2 hidden layers of 128 and 256 neurons.

We were able to create a CNN model with a train:test ratio of 0.75:0.25, however when we visualised the training progress of the model against epochs, we found that the accuracy had plateaued after just around 6 epochs. This was a sign of overfitting and prompted us to improve the model.

We decided to then use an additional dropout layer, which randomly drops 20% of the neurons in the 256 layer to reduce dependency on each neuron which will hopefully reduce overfitting.
## Conclusion
## Contributors
## References
- https://www.baeldung.com/cs/ml-relu-dropout-layers
- https://www.datacamp.com/tutorial/cnn-tensorflow-python
- https://www.kaggle.com/code/prashant111/random-forest-classifier-feature-importance
- https://www.youtube.com/watch?v=bDhvCp3_lYw
- https://seaborn.pydata.org/generated/seaborn.violinplot.html
- https://seaborn.pydata.org/generated/seaborn.catplot.html

