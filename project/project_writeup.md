# Enron Fraud Detection
### Machine Learning Project - Udacity Data Analyst Nanodegree
### Project Writeup
----


[//]: # (Image References)
[image1]: ./img/boxplots_1.png
[image2]: ./img/boxplots_2.png
[image3]: ./img/boxplots_3.png
[image4]: ./img/boxplots_4.png
[image5]: ./img/boxplots_new_features.png


## 1. Presentation of project

This project aims at predicting whether a person involved in the Enron demise is a person of interest (POI). This prediction uses financial data (salary, incentives, stock value etc.) and email data (number of emails sent/received, number of emails to/from persons of interest etc.). A person of interest (POI) is someone who has been convicted, heard as a witness or otherwise involved in the judicial proceedings that followed the Enron scandal.

**Note:** The full record of emails for each person is also provided and could be a useful addition to the dataset. Processing this data involves common Natural Language Processing techniques, but I have elected not to use them in this project due to time constraints. The sheer amount of data in these emails would have made the whole project a lot more demanding in terms of time requirements. For an example of NLP project, see my [Shinyapp](https://lucfrachon-ds.shinyapps.io/NLP_NextWordPrediction/).

The dataset contains 146 observations (persons) and 21 features for each. A detailed exploratory analysis of the dataset including plots can be found in [`enron_exploration.html`](./enron_exploration.html) (a render of `enron_exploration.ipynb`). `poi` is the outcome variable (that we are trying to predict) and there are 20 predictors. NAs in the dataset are actually zeros; here is a count of NAs by feature:

|Feature name		   | NAs |
|--------------------------|-----|
|loan_advances             |  142|
|director_fees             |  129|
|restricted_stock_deferred |  128|
|deferral_payments         |  107|
|deferred_income           |   97|
|long_term_incentive       |   80|
|bonus                     |   64|
|from_poi_to_this_person   |   60|
|shared_receipt_with_poi   |   60|
|to_messages               |   60|
|from_this_person_to_poi   |   60|
|from_messages             |   60|
|other                     |   53|
|expenses                  |   51|
|salary                    |   51|
|exercised_stock_options   |   44|
|restricted_stock          |   36|
|total_payments            |   21|
|total_stock_value         |   20|
|email_address             |    0|
|poi                       |    0|


`loan_advances` has 142 NAs out of 146 observations, so we will not retain it in our model. 

After these investigations, replaced them with 1.e-5 to avoid issues with log conversions (as log(0) is undefined). For the same reason, I took the absolute values of all variables (I noticed that two of the variables, `deferred_income` and `restricted_stock_deferred`, had only negative values whereas all the other were always positive).

There are two outliers immediately visible: `LOCKART EUGENE E` and `THE TRAVEL AGENCY IN THE PARK`. The first one has zeros (or 1.e-5 now) on every single feature, and the latter is obviously not a real person but a company. I decided to retain the first one (after all, the fact that this Mr. Lockart received nothing and is not a POI might be interesting per se), and drop the second one as we are investigating people, not suppliers.

By plotting the first two predictors, I found a clear outlier whose salary seemed almost an order of magnitude above everyone else's. Upon closer look, it appeared that this was actually the 'TOTAL' row from the `enron61702insiderpay.pdf` document! I quickly dropped this entry. The other outliers are actual data and I therefore decided to retain them.

We are therefore left with 144 observations.

Further exporation also allowed me to notice that for two of the individuals in the dataset, the data had been shifted to the left or to the right by one column. The persons in question are Robert Belfer and Sanjay Bhatnagar. Again, I corrected the data by shifting everything back to its correct position.


## 2. Feature selection and engineering

The steps below are described in detail in [`enron_exploration.html`](./enron_exploration.html), along with the plots I mention.
To decide which variables to retain, I made boxplots of each grouped by POI to see which displayed significant association with the outcome variable. By plotting histograms I also checked which features would benefit from logarithmic transformation.

I have reproduced the boxplots from the exploration file below to illustrate what the reasoning was behind the variable selection process.

* Financial information:

![alt_text][image1]
![alt_text][image2]


* Stock information:

![alt_text][image3]

* E-mail information:

![alt_text][image4]


Finally, I aggregated email features to reduce dimensionality. Given the low number of observations, it is important to make the dataset as low-dimensional as possible to make ML algorithms work. The resoning behind these new variables is the following:

- `sent_vs_received` = `from_messages` / `to_messages`: I had an intuition that because POIs are probably mostly senior executives, they would tend to receive much more emails than they send out. This variable reflects this.
- `total_emails` = `from_messages` + `to_messages`: Again, I would expect POIs to be involved (even if just CC'd) in much more email traffic than the average.
- `emails_with_poi` = `from_this_person_to_poi` + `from_poi_to_this_person` + `shared_receipt_with_poi`: If POIs are a somewhat specific population within the company (and hopefully they are, otherwise our attempts at a predictive model are doomed to failure), it is reasonable to think that they are involved in the same conversations, whether they send or receive the emails. Aggregating these three variables reflects this intuition and reduces the number of features.

* Here is how these new variables look in a boxplot:

![alt_text][image5]

To determine the optimal set of variables, I explored the feature space along 4 dimensions:

- Apply log transform to the variables that most benefit from it (from the exploratory analysis plots) or keep original values
- Drop `deferral_payments`or keep it. This feature has a high number of zeros (107 out of 146 observations) and the boxplot shows that there does not seem to be a clear separation between POIs and non-POIs. Note that we drop `loan_advances` in all model iterations because it has 142 zeros. Converserly, `director_fees` and `restricted_stock_deferred` are also near-zero variables but they both have an interesting property: Only non-POIs have non-zero values. Therefore, they seem important to improve the model's specificity. They are therefore retained in all model iterations.
- Retain original e-mail features or replace them with the new features described above.
- Drop or retain `total_payment` as it is an aggregate of all other salary features; drop or retain `exercised_stock_options` and `restricted_stock` as we already have `total_stock_value` which is the sum of these variables (plus `restricted_stock_deferred`, which we have decided to keep anyway as explained above). You might notice that I'd rather keep `total_stock_value` and drop its components, while I chose to do the opposite with `total_payment`. This is because there are only three variables that make up `total_stock_value`, therefore I feel that we'd loose less information by using the aggregate than if we did so with `total_payment`, which contains values from 9 features.

I also accidentally noticed that keeping both `bonus` and its log-transformed variant helps model performance significantly. This is somewhat counter-intuitive, as it is the same information that is encoded in both, albeit differently, but the model's  F1-score drops when removing one or the other.

With this in mind, I tried many iterations of the model. I used cross-validation to tune them and made note of the best cross-validation F1-score. Here is the outcome (best cross-validation score in **bold**):

| Log transforms | Keep `deferral_payments` | Use new email features | Keep `total_payments`/stock vars | RF score | SVC score  |
|----------------|--------------------------|------------------------|----------------------------------|----------|------------|
| No		 | Yes			    | No		     | Yes				| 0.2491   | 0.1835	|
| Yes		 | Yes			    | No		     | Yes				| 0.2422   | 0.4037	|
| Yes		 | No			    | No		     | Yes				| 0.2717   | 0.3757	|
| Yes 		 | Yes			    | Yes		     | Yes				| 0.2510   | 0.4046	|
| Yes		 | Yes			    | No		     | No				| 0.2661   | 0.5252	|
| Yes		 | No			    | No		     | No				| 0.2443   | 0.5122	|
| Yes		 | Yes			    | Yes		     | No				| 0.2956   | **0.5671**	|
| Yes		 | No			    | Yes		     | No				| 0.2763   | 0.5655	|



The final set of variables that I used is therefore:

- `salary`
- `bonus`
- `expenses`
- `director_fees`
- `log_deferred_income`
- `log_long_term_incentive`
- `log_other`
- `log_restricted_stock_deferred`
- `log_bonus`
- `log_total_stock_value`
- `sent_vs_received`
- `total_emails`
- `emails_with_poi`


As part of the pipeline used to train the model, I scaled the variables. I tried both RobustScaler and StandardScaler and the latter performed better.


## 3. Machine Learning algorithm

Given the time constraint, I wanted to try a fairly simple, standard algorithm and an ensemble algorithm. For the former, I selected a Support Vector Machine classifier because although fairly simple and easy to optimize, it gives good results in many situtations. As for the latter, I chose Random Forest because it is relatively easy to tune with a limited number of really impacting parameters, and it worked well for me in several projects in the past.

The detailed iterations of the model are presented in the section Against expectations, the SVC performed much better after tuning and training:

**Cross-validation accuracy (Stratified Shuffle Split, 10 folds):**

| Model         | Best CV F1-score |
|---------------|------------------|
| SVC           | 0.5671           |
| Random Forest | 0.2956           |


## 4. Algorithm tuning

Algorithm tuning is critical because each dataset is different. All algorithms have a number of "knobs" that can be tuned to improve performance for the particular data we are dealing with. The 'No Free Lunch' theorem tells us that no model can be the best performer accross all possible datasets, which is why trial and error is still very much the norm in Machine Learning. Failure to tune a model might lead to its rejection, even though it had the potential to perform well. In the worst case, it might even prevent a model from converging.

The `tester.py` module used to grade the project performs 1000 stratified splits. Tuning algorithms this way would be too time-consuming so I decided to proceed in two steps:

- Use grid search with 10-fold startified shuffle split cross-validation (using 30% of the data for validation) to discover the best parameters
- Use these best parameters to create a pipeline that scales features and fits a model, to be passed to `tester.py`

For the SVC model, I varied `C` and `gamma` on a log space using base 2, i.e. a grid where values are evenly separated on a log2 scale. Values ranged from 1.95e-3 to 32, with 15 steps.

For the Random Forest, I varied `n_estimators` and `max_features` using the values (3, 4, 5, 8) and (2, 3, 4, 5) respectively (earlier tests allowed me to zoom in to these ranges).


## 5. Validation strategy

In an ideal world, we want to train, tune and test the model on separate parts of the data. The process would go like this:

- Separate the data into 3 parts (60% / 20% / 20% for instance) -- the training, validation and test sets
- Fit a model on the training set and evaluate the prediction performance on the validation set
- Adjust parameters, features, scaling methods etc. and repeat the previous step until performance no longer improves
- Test the final model's predictive performance on the held-out test set

With a smaller dataset, or if we are particularly warry of overfitting, we might be reluctant to cut the data into 3 parts so we simply split it in two (70% / 30% for instance). These are our training and test sets. However we still need to tune the model and to do that, to assessing each iteration's performance on a separate dataset to the one used to fit it. This is where cross-validation comes into play. For instance, cross-validation on stratified shuffled splits with 100 splits and test size = 0.3 would go like this:

- 30% of the dataset is designated as a test set, the other 70% are the training set. The 30% are selected at random while ensuring that the class proportions are the same in the training and test sets.
- The model is fitted on the training set and evaluated on the test set.
- The previous two steps are repeated 99 times (giving us 100 stratified random training and test sets) and the 100 scores are averaged
- The whole process is repeated for the next set of parameter values
- The set of parameters that leads to the best average score is retained. We have found our optimal model.

This method simulates a much larger dataset than is actually available and is less prone to overfitting than trying different sets of parameters on exactly the same data. Once our optimal parameters are found and the model is trained, we can then test it against the held-out test set to get an estimate of the real-world performance.

In both the cases described above, tuning and testing repeatedly on the training set only would likely lead to strong training accuracy, but the model would be unlikely to generalize well to unseen observations. Likewise, repeatedly evaluating performance on the test set during the tuning phase would lead to a model that would be effective on the test set, but not necessarily on new data.

In the case of the Enron dataset, things are slightly different. This dataset is particular because it has very few observations (146). If we were to split it into training and test sets, we simply would not have enough data to reliably train a statistical model. This is why we only use one dataset. To try and prevent overfitting, `tester.py` uses many stratified random splits.

In the tuning phase, we run a grid search to search for optimal values through the parameter space. To do this, we use cross-validation exactly as described above. This gives us the tuning parameters to use in the second phase. We use 100 splits for each parameter set (my tests show that the parameter set retained by the grid search does not change much past 50 splits). I will discuss which performance metrics I used in the next chapter.

The optimal parameters are then saved and passed to `tester.py`. This function creates 1,000 stratified folds, each with a 10% test set, and fits the saved model to each. In each fold, the locally trained model's predictions on the test set are then compared with the labels and true positives, true negatives, false positives and false negatives are counted.

These four values are summed up for all 1,000 folds to provide the final performance metrics.

### 6. Final model's performance

Another peculiarity of this dataset is its unbalanced nature. Only 13% of the individuals are actually POIs, therefore using accuracy as the performance metric to discover the optimal model parameters is not a good approach: In the extreme case, a model classifying everybody as non-POI would reach 87% accuracy!

This is why I used the F1-score. This is a composite score that balances precision and recall (it is their [harmonic mean](https://en.wikipedia.org/wiki/F1_score)). There is necessarily a trade-off between precision and recall and the F1-score is a way to try and reach a good compromise between the two.

The final model is a Support Vector Classifier with C = 8.0 and gamma = 0.5.

* Confusion matrix:

|          label |   0 |  1 |
|----------------|-----|----|
| **prediction** |   - |  - |
|          **0** |12801| 959|
|          **1** |199  |1041|

* Metrics:

	- Accuracy: 0.92280
	- Precision: 0.83952
	- Recall: 0.52050
	- F1: 0.64259
	
Concretely, this means that:

- 92.3% of all observations where correctly classified, a good improvement over the 87% that the no-information model would reach
- 84.0% of the individuals classified as POI where actually POIs -- this is also called the Predictive Positive Value. In other words, when the model tells us that someone is a POI, we are 84% certain that it is right.
- 52.1% of the actual POI where classified as POIs. This is also called Sensitivity. In other words, given a bunch of people, we estimate that the model will correctly detect 52.1% of the POIs.

A possible way to play with this trade-off would be to have the model compute class probabilities and set a threshold manually for class 1 vs. class 0. By adjusting this threshold, we would be able to increase or decrease precision vs recall, depending on whether we are more interested in detecting as many POIs as possible even if some non-POI get flagged, or be conservative and avoid false alarms even if that means missing some actual POIs.