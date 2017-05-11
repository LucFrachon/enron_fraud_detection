# Enron Fraud Detection
### Machine Learning Project - Udacity Data Analyst Nanodegree
### Project Writeup
----


## 1. Presentation of project

This project aims at predicting whether a person involved in the Enron demise is a person of interest (POI). This prediction uses financial data (salary, incentives, stock value etc.) and email data (number of emails sent/received, number of emails to/from persons of interest etc.). A person of interest (POI) is someone who has been convicted, heard as a witness or involved in the judicial proceedings that followed the Enron demise.

**Note:** The full record of emails for each person is also provided and could be a useful addition to the dataset. Processing this data involves common Natural Language Processing techniques, but I have elected not to use them in this project due to time constraints. The sheer amount of data in these emails would have made the whole project a lot more demanding in terms of time requirements. For an example of NLP project, see my [Shinyapp](https://lucfrachon-ds.shinyapps.io/NLP_NextWordPrediction/).

The dataset contains 146 observations (persons) and 21 features for each. A detailed exploratory analysis of the dataset including plots can be found in `enron_exploration.html` (a render of `enron_exploration.ipynb`). `poi` is the outcome variable (that we are trying to predict) and there are 20 predictors.

At some point in my exploration I realized that for two individuals, data had been shifted left or right by one column, compared to the contents of the `enron61702insiderpay.pdf`
By plotting the first two predictors, I found a clear outlier whose salary seemed almost an order of magnitude above everyone else's. Upon closer look, it appeared that this was actually the 'TOTAL' row from the pdf document! I quickly dropped this entry.

Further exporation also allowed me to notice that for two of the individuals in the dataset, the data had been shifted to the left or to the right by one column. The persons in question are Robert Belfer and Sanjay Bhatnagar. Again, I corrected the data by shifting everything by one column.

