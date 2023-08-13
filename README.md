# ML-Programming-Final-Project
Hierarchical Attentional Hybrid NeuralNetworks and XLNet for document classification.
*********************************************************************************************************************************************************************************
## Paper Title and Link :
HAHNN: Hierarchical Attentional Hybrid NeuralNetworks for document classification
github link : [https://github.com/luisfredgs/cnn-hierarchical-network-for-document-classification](link)

## Description of Paper
*********************************************************************************************************************************************************************************
RMDL model can be seen as ensemble approach for deep learning models.RMDL solves the problem of finding the best deep learning structure and architecture while simultaneously improving robustness and accuracy through ensembles of deep learning architectures.

## Context of the Problem
********************************************************************************************************************************************************************************
* The continually increasing number of complex datasets each year necessitates ever improving machine learning methods for robust and accurate categorization of these data.
* General documentation classification models not take account of the context.
* Generally, deep learning models needs lots of computational power to do the mathematical functions
* Users need to have high configuration hardware resources for train a model from scratch to this kind of big size data
* So, They proposed an hierarchical attention model based approach for language tasks instead of RNN.

## Implementation 
*********************************************************************************************************************************************************************************
* Try to replicate the results given in paper on document classification datasets with HAHNN model. [a link](https://github.com/rubaramanan/ML-Programming-Final-Project/blob/main/hahnn/cnn-hierarchical-network-for-document-classification/hahnn-for-document-classification.ipynb)
* choose imdb datset as standard dataset.
* To compare the performance of HAHNN model, trained XLNet model on the above-mentioned dataset. [a link](https://github.com/rubaramanan/ML-Programming-Final-Project/blob/main/src/Movie_Reviews_XLNet.ipynb)

## Results and Conclusion
*********************************************************************************************************************************************************************************
For Results each dataset is trained with HAHNN and XLNet model and comparison is present in the form of accuracy, f1-macro and f1-weighted. HAHNN paper only shows accuarcy in their results however I feel f1-score should be a better metric to assess the performance of document classification tasks.

* Both the models perform equally well
* XLNet model performs well however the execution time was 6+ hours with four epoches, if epoches 8, 13+ hours.

|Models|Accuracy|
|------|--------|
|HAHNN|0.98|
|XLNet|0.92|