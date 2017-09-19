# insuranceqa-pytorch
Partial Implementation of LSTM based Deep Learning Models for Non-Factoid Answer Selection by Ming et al.

Model includes only a training graph without attention and with cosine similarity.
TODO:
1) Create a testing graph.
2) Implement attention.
3) Use GESD similarity instead of cosine.

I referred https://github.com/codekansas/keras-language-modeling and https://github.com/white127/insuranceQA-cnn-lstm for porting the implementation.

Link to dataset: https://github.com/codekansas/insurance_qa_python

To run:
1) Clone the dataset in the directory where model.py is present.
2) Run model.py
