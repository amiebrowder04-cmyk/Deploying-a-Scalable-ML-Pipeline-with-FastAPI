# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model uses a random Forest Classifier to predict whether an indavidual's annual income is greater than $50k or less than or equal to $50k. The model was developed as part of a machine learning pipeline using census demographic and employment data.

## Intended Use

The model is intended for educational and analytica purposes to demonstrate a machine learning pipeline for predicting income categories from census data. It should not be used to make decisions about an indavidul's employment, credit, housing, insurnace, or other high impact oppertunities. 

## Training Data

The model was trained using the cencus.csv dataset. The data contains demographic and employment related information, including age, education, ocupation, workingclass, marital status, relationship, race, sex, hours worked per week, and other cencus features, Categorical features were processed using one-hot encoding, and the income label was converted to a binary value representing >50k or <=50k income.

## Evaluation Data

The data was divided into training and evaluation sets using an 80/20 train-test split with a random state of 42. he evaluation set was not used to train the model and was used to measure model preformance.

## Metrics

The model was evaluated using precision, recall, and F1 score.
-Precision: 0.7419
-Recall: 0.6384
-F1 score: 0.6863

## Ethical Considerations

The model uses deomographic and socioeconomical characteristics that may be associated with historical inequalities or biases in the underlying data. Features such as race, and sex may contribute to differences in model preformance across groups. 

## Caveats and Recommendations

Model preformance may vary across demographic and categorical groups. Some groups in the evaluation data have relativly few observations, which can make their preformance metrics less reliable. The model was trained and evaluated on a specific census dataset and may not generalize to other opoulations or current economic conditions. 

The model should be used for educationl and analytical purposes ratehr than high-impact indavidule decision-making. Futher improvements could include additional data, more extensive model evaluation, fairness analysis across demographic groups, and hyperpearameter tuning. 
