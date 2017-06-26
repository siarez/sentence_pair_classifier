This is a quick and dirty Tensorflow implementation of an attention based NLP model that learns relationships between sentences.
It is inspired by the paper: [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/pdf/1606.01933v1.pdf)

The orginal paper has used the model on the Stanford Natural Language Inference (SNLI) dataset. I adapted this model to compete in [Quora Question Pairs on Kaggle](https://www.kaggle.com/c/quora-question-pairs)

There are three main files:
1. **pair_classifier_model.py** defines a class that create the model's graph. The paper has not described their model in complete detail. So it may differ in some ways, but I believe it captures the essence of what was describe in the paper. 
    
2. **pair_classifier_train.py** loads and prepares training and validation datasets and iterates through question pairs one by one. It also saves model checkpoints in ./save directory, and produces a log on ./log for viewing in Tensorboard. No batching has been implemented. 
 
3. **pair_classifier_infer.py** loads and prepares test data for the competition. It then restores the model from the latest checkpoint created by *pair_classifier_train.py* and iterates through the test data. Finally, it creates two CSV files ready to be submitted to Kaggle. 

For a detailed description of this project and the results, please see [this post](http://www.siarez.com/projects/quora-question-pairs). 