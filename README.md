# README
---

## Enron Fraud Detection

This project aims at predicting whether a person involved in the Enron demise is a person of interest (POI). This prediction uses financial data (salary, incentives, stock value etc.) and email data (number of emails sent/received, number of emails to/from persons of interest etc.). A person of interest (POI) is someone who has been convicted, heard as a witness or otherwise involved in the judicial proceedings that followed the Enron scandal.

## Files

* `enron61702insiderpay.pdf`: A document that contain all the financial information about the employee in the dataset, provided for reference

* `/project`: Folder containing all the code and pickle files for the project
	- `enron_exploration.ipynb`: Jupyter Notebook detailing the exploratory analysis of the dataset
	- `enron_exploration.html`: Same content rendered in HTML
	- `poi_names.py`: Main code file; this is the file to execute to define and tune the model.
	- `prep_dataset.py`: Code to prepare the data, clean it and define new features.
	- `ml_models.py`: Code containing functions to tune an SVM classifier and a Random Forest classifier, and define sklearn pipelines
	- `tester.py`: Code used to fit the retained model on 1000 stratified folds of the data and evaluate performance. Execute from command line after `poi_names.py`.
	- `final_project_dataset.pkl`: Raw dataset (as a dictionary), loaded by `poi_names.py`
	- `my_classifier.pkl`: Pickle dump of the tuned classifier pipeline
	- `my_feature_list.pkl`: Pickle dump of the list of features retained in the final model.
	- `my_datast.pkl`: Final dataset (cleaned and containing engineered features)
	- `project_writeup.md`: A short project report where I present the steps taken to build, train and test the model.
	
* `/tools`: Folder containg tools for parsing emails and feature dictionaries. In the context of this project, only `feature_format.py` is used.


