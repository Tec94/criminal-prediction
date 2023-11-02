import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

data = pd.read_csv("./criminal-prediction/compas-scores-two-years.csv", header=0)
df = data.drop(labels=['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'days_b_screening_arrest',
                         'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'c_days_from_compas',
                         'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc',
                         'r_jail_in', 'r_jail_out', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'decile_score.1',
                         'violent_recid', 'vr_charge_desc', 'in_custody', 'out_custody', 'priors_count.1', 'start', 'end',
                         'v_screening_date', 'event', 'type_of_assessment', 'v_type_of_assessment', 'screening_date',
                         'score_text', 'v_score_text', 'v_decile_score', 'decile_score', 'is_recid', 'is_violent_recid'], axis=1)
df.columns = ['sex', 'age', 'age_category', 'race', 'juvenile_felony_count', 'juvenile_misdemeanor_count', 'juvenile_other_count',
              'prior_convictions', 'current_charge', 'charge_description', 'recidivated_last_two_years']

value_counts = df['charge_description'].value_counts()
df = df[df['charge_description'].isin(value_counts[value_counts >= 70].index)].reset_index(drop=True) # drop rare charges
for colname in df.select_dtypes(include='object').columns:
   one_hot = pd.get_dummies(df[colname])
   df = df.drop(colname, axis=1)
   df = df.join(one_hot)
y_column = 'recidivated_last_two_years'
X_all, y_all = df.drop(y_column, axis=1), df[y_column]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3)

X_caucasian = X_test[X_test['Caucasian'] == 1]
y_caucasian = y_test[X_test['Caucasian'] == 1]
X_african_american = X_test[X_test['African-American'] == 1]
y_african_american = y_test[X_test['African-American'] == 1]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Training accuracy:", model.score(X_train, y_train))
print("Testing accuracy:", model.score(X_test, y_test))

y_pred_caucasian = model.predict(X_caucasian)
print("Caucasian acceptance rate:", np.count_nonzero(y_pred_caucasian) / len(y_pred_caucasian))
y_pred_afam = model.predict(X_african_american)
print("African-American acceptance rate:", np.count_nonzero(y_pred_afam) / len(y_pred_afam))

y_pred_caucasian = model.predict(X_caucasian)
y_pred_afam = model.predict(X_african_american)
print("Class 0 calibration, Caucasian:", np.count_nonzero(1 - y_pred_caucasian[(y_caucasian == 1)]) / np.count_nonzero(1 - y_pred_caucasian))
print("Class 1 calibration, Caucasian:", np.count_nonzero(y_pred_caucasian[y_caucasian == 1]) / np.count_nonzero(y_pred_caucasian))
print("Class 0 calibration, African-American:", np.count_nonzero(1 - y_pred_afam[y_african_american == 1]) / np.count_nonzero(1 - y_pred_afam))
print("Class 1 calibration, African-American:", np.count_nonzero(y_pred_afam[y_african_american == 1]) / np.count_nonzero(y_pred_afam))