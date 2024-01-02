# Income Classification using Logistic Regression

In this project, I will use a dataset containing census information from the 1994 Census to create a logistic regression model, that predicts whether or not a person makes more than $50,000 a year.


### Dataset

Data set is available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/20/census+income).

### Features

Input and Output `features`:
* age: continuous
* workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
* education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
* race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
* sex: Female, Male
* capital-gain: continuous
* capital-loss: continuous
* hours-per-week: continuous
* native country: discrete
* income: discrete, >50K, <=50K

------

### EDA and Logistic Regression Assumptions
1. The dataset has been saved as a dataframe named `df`. The outcome variable here is `income`. Check if the dataset is `imbalanced`.
2. Notice we have created a variable named `feature_cols`. This contains a list of the variables we will use as our predictor variables.
    `Transform` the dataset of predictor variables to dummy variables and save this in a new DataFrame called `X`.
3. Using `X`, create a `heatmap` of the correlation values.
4. Determine if `scaling` is needed for `X` prior to modeling. Then create the `y` output variable which is binary, `0` when income is less than $50K, `1` when greater than $50K.

### Logistic Regression Models and Evaluation
5. Split the data into a training and testing set. Set the `random_state` to 1 and `test_size` to `.2`.
6. Print the model parameters (`intercept` and `coefficients`).
7. Evaluate the predictions of the model on the `test set`. Print the `confusion matrix` and `accuracy score`.
8. Create a new DataFrame of the model coefficients and variable names. Sort values based on coefficient and exclude any that are equal to zero. Print the values of the DataFrame.
9. Create a `barplot` of the coefficients sorted in ascending order.
10. Plot the `ROC curve` and print the `AUC value`.