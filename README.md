# MachineLearningSamples

This repo hosts variety of examples based on Apache Spark MLIB.

## Databricks Notebooks

### [Decision Tree](https://github.com/aosama/MachineLearningSamples/blob/master/databricks/DecisionTreeShapeExample.scala)

## Scala IDE Based Examples

### [Decision Tree](https://github.com/aosama/MachineLearningSamples/blob/master/src/main/scala/org/ibrahim/ezmachinelearning/DTShapeTypeExample.scala)
A vanilla decision tree example.

### [Decision Tree with Stratified Sampling](https://github.com/aosama/MachineLearningSamples/blob/master/src/main/scala/org/ibrahim/ezmachinelearning/DTShapeTypeStratifiedExamples.scala)
How to get a stratified sample so the test and train datasets are sampled accross possible values.

### [Decision Tree with Categorical Feature in the DataSet](https://github.com/aosama/MachineLearningSamples/blob/master/src/main/scala/org/ibrahim/ezmachinelearning/DTShapeTypeWithCategoricalFeaturesExample.scala)
How to index and encode categorical features.

### [Decision Tree Multiple Categorical and Continuous Features in the DataSet](https://github.com/aosama/MachineLearningSamples/blob/master/src/main/scala/org/ibrahim/ezmachinelearning/DTCensusIncomeExample.scala)
How to handle multiple categorical and continuous features on a real-life data set.
Uses the Census Income data set.

### [Random Forest Multiple Categorical and Continuous Features in the DataSet](https://github.com/aosama/MachineLearningSamples/blob/master/src/main/scala/org/ibrahim/ezmachinelearning/RFCensusIncomeExample.scala)
How to handle multiple categorical and continuous features on a real-life data set.
Uses the Census Income data set.

## Data Sets References

#### [Census Income DataSet](http://archive.ics.uci.edu/ml/datasets/Census+Income)
First line from adult.test file removed for loading into Spark.

Census Income data set citation:
Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

#### [Default of Credit Card Clients DataSet](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
<br/> 
X2: Gender (1 = male; 2 = female).
<br/> 
X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
<br/> 
X4: Marital status (1 = married; 2 = single; 3 = others).
<br/> 
X5: Age (year).
<br/> 
X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
<br/> 
X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.
<br/> 
X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005. 
<br/>