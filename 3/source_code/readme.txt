All of this was run using Ubuntu on Windows in a Python virtual environment.

1. Retrieve the dataset from Kaggle (https://www.kaggle.com/c/nlp-getting-started/data).
2. Place test.csv and train.csv into a folder named "Data" in the same folder as the python files.
3. Create a folder inside "Data" named "Processed"
4. Run the following commands to install the required Python modules:
pip install pytorch
pip install matplotlib
pip install pandas
pip install torch
pip install sklearn
pip install torchtext
pip install transformers
pip install seaborn

Because we utilized multiple models with different preprocessing steps, we split our code based upon the models themselves.

To run the data exploration, open exploring.ipynb in a Jupyter notebook.

For the models, there are .py files and .ipynb files to go with each. Each are named the same as their corresponding .py files. The Python notebooks display visualizations of model metrics. It is recommended that you utilize the Python notebooks as they display more information than just predicting labels for the test file.

To run the Random Forest, Support Vector Machine, Multinomial Naive Bayesian, Linear Regression, and Decision Tree classifiers, run the command:
python3 Naive_Bayes_and_Others/Naive_Bayes_and_Others.py

To run the BERT model from the source_code folder, run the commands: 
python3 "bert/bert pretrain.py"
python3 "bert/bert test.py"
python3 "bert/bert train model.py"

To run the LSTM model from the source_code folder, run the command:
python3 lstm/lstm.py