import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
df_01 = pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')
df = df_01.copy()
df
st.subheader("Pages")

    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"
      # Dataset
 if st.checkbox("Dataset"):
	if st.button("Head"):
		st.write(df.head(4))
	if st.button("Tail"):
		st.write(df.tail())
	if st.button("Infos"):
		st.write(df.info())
	if st.button("Shape"):
		st.write(df.shape)
	else:
		st.write(df.head(2))
    # EDA
gj = df.groupby(['y', 'job'])['job'].count()
gj = gj.unstack(level=0)
gj.plot(kind='bar', figsize=(9,8), use_index=True,)
plt.show(gj)
gm = df.groupby(['y', 'marital'])['marital'].count()
gm = gm.unstack(level=0)
gm.plot(kind='bar', figsize=(9,8), use_index=True,)
plt.show(gm)

ge = df.groupby(['y', 'education'])['education'].count().transform(lambda x: x/x.sum())
ge = ge.unstack(level=0)
ge.plot(kind='bar', figsize=(9,8), use_index=True,)
plt.show(ge)

gd = df.groupby(['y', 'default'])['default'].count()
gd = gd.unstack(level=0)
gd.plot(kind='bar', figsize=(9,8), use_index=True,)
plt.show(gd)

g = df.groupby(['y', 'month'])['month'].count()
g = g.unstack(level=0)
g.plot(kind='bar', figsize=(9,8), use_index=True,)
plt.show(g)

gh = df.groupby(['y', 'housing'])['housing'].count()
gh = gh.unstack(level=0)
gh.plot(kind='bar', figsize=(9,8), use_index=True,)
plt.show(gh)

gc = df.groupby(['y', 'contact'])['contact'].count()
gc = gc.unstack(level=0)
gc.plot(kind='bar', figsize=(9,8), use_index=True,)
plt.show(gc)

gl = df.groupby(['y', 'loan'])['loan'].count()
gl = gl.unstack(level=0)
gl.plot(kind='bar', figsize=(9,8), use_index=True,)
plt.show(gl)
gdow = df.groupby(['y', 'day_of_week'])['day_of_week'].count()
gdow = gdow.unstack(level=0)
gdow.plot(kind='bar', figsize=(9,8), use_index=True,)
plt.show(gdow)
gpo = df.groupby(['y', 'poutcome'])['poutcome'].count()
gpo = gpo.unstack(level=0)
gpo.plot(kind='bar', figsize=(9,8), use_index=True,)
plt.show(gpo)
if st.button("EDA"):
	st.subheader("EDA visualisations")
	st.text("Contruite avec Streamlit")
	st.text("Thanks to the Streamlit Team Amazing Work")
  if st.button("Job"):
		st.write(plt.show(gj))
	if st.button("Marital"):
		st.write(plt.show(gm))
	if st.button("Education"):
		st.write(plt.show(ge))
	if st.button("default"):
		st.write(plt.show(gd))
  if st.button("month"):
		st.write(plt.show(g))
  if st.button("housing"):
		st.write(plt.show(gh))
  if st.button("contact"):
		st.write(plt.show(gc))
  if st.button("loan"):
		st.write(plt.show(gl))
  if st.button("day_of_week"):
		st.write(plt.show(gdow))
  if st.button("poutcome"):
		st.write(plt.show(gpo))
  st.write(sns.boxplot(df, x='y', y='age')
plt.show
plt.figure(figsize=(10,8))
sns.displot(df['age']))
  st.write(sns.boxplot(df, x='y', y='duration')
plt.show
plt.figure(figsize=(10,8))
sns.displot(df['duration']))
 st.write(sns.boxplot(df, x='y', y='campaign')
plt.show
plt.figure(figsize=(10,8))
sns.displot(df['campaign']))
  st.write(sns.boxplot(df, x='y', y='pdays')
plt.show
plt.figure(figsize=(10,8))
sns.displot(df['pdays']))
st.write(print(f"Duplicate values count: {df.duplicated().sum()}"))
st.write(print(f"Missing values count: {df.isna().sum().sum()}"))
st.write(df.loc[df.duplicated(keep=False)].head(6))
df = df.drop_duplicates()
columns_numerical = [
    'age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed'
]
columns_categorical = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
    'poutcome', 'y'
]
st.write(df.describe())
columns_continuous = [
    'age', 'duration', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
]

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

axes = axes.flatten()

for i, num in enumerate(columns_continuous):
	axes[i].hist(df[num], bins=30, alpha=0.7, edgecolor='black')
	axes[i].set_title(f'{num} ')

for i in range(len(columns_continuous), len(axes)):
	fig.delaxes(axes[i])

st.write(plt.tight_layout())
st.write(plt.show())
#Data cleaning
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
df.drop(columns='duration', inplace=True)
df.drop(columns='pdays', inplace=True)
X = df.drop(columns='y')
y = df['y']
if st.button("Data cleaning"):
	st.subheader("Preprocessing")
  st.write(print(f"{X.shape = }"))
  st.write(print(f"{y.shape = }"))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, 
    stratify=y,  
    random_state=42 
)

st.write(print(f"{X_train.shape = }"))
st.write(print(f"{X_test.shape = }"))
st.write(print(f"{y_train.shape = }"))
st.write(print(f"{y_test.shape = }"))
check_stratification = pd.df(
    {
        'y_train': y_train.value_counts(normalize=True),
        'y_test': y_test.value_counts(normalize=True)
    }
)
st.write(check_stratification)
X_train[['campaign', 'previous']] = X_train[['campaign', 'previous']].apply(lambda x: np.log1p(x))
st.write(X_train)
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
X_train['month'] = X_train['month'].map(month_mapping)

day_mapping = {
    'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5
}
X_train['day_of_week'] = X_train['day_of_week'].map(day_mapping)


def cyclic_encode(df, col, max_val):
	df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
	df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
	
	return df

st.write(cyclic_encode(X_train, 'month', 12))
st.write(cyclic_encode(X_train, 'day_of_week', 5))
columns_ohe = ['job', 'marital', 'education', 'default', 'housing' ,'loan', 'contact', 'poutcome']
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
X_train_ohe = ohe.fit_transform(X_train[columns_ohe])
X_train = pd.concat([X_train, X_train_ohe], axis=1).drop(columns=columns_ohe)
st.write(X_train.head(3))
st.write(columns_standard_scale = [
    'age', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
]

standard_scaler = StandardScaler().set_output(transform='pandas')
X_train_standard_scaled = standard_scaler.fit_transform(X_train[columns_standard_scale])
X_train[columns_standard_scale] = X_train_standard_scaled
X_train.head(3))
y_train = y_train.map({'no': 0, 'yes': 1})
#Machine learning
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
if st.button("Machine learning"):
	st.subheader("Model Arquitecture")
  class ModelWrapper:
    def __init__(self, model_class, seed=42, params=None, scorer=None):
        if params is None:
            params = {}
        params['random_state'] = seed  
        self.model = model_class(**params)
        # Feature importances for tree-based models
        self.feature_importances_ = None
         self.coef_ = None  
        self.intercept_ = None  
        # Weights for models like SVR (Support Vector Regressor)
        self.dual_coef_ = None  # Placeholder for dual weights
        # Place holders for model evaluation
        self.scorer = scorer  
        self.best_params_ = None  # To store the best model's parameters

    def fit(self, X, y):
       self.model.fit(X, y)

        # If the model has feature_importances_ (for tree-based models)
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_
        
        # For models like LinearRegression, Ridge, Lasso, etc., use coefficients
        elif hasattr(self.model, "coef_"):
            self.coef_ = self.model.coef_

        elif hasattr(self.model, "intercept_"):
            self.intercept_ = self.model.intercept_
        
        # For models like SVR (Support Vector Regressor), check if coefficients exist
        elif hasattr(self.model, "dual_coef_"):
            self.dual_coef_ = self.model.dual_coef_
          return self

    def predict(self, X):
      return self.model.predict(X)

    def score(self, X, y):
      if self.scorer:
            # Use custom scorer if provided
            return self.scorer(self.model, X, y)  # Assuming scorer takes the model, X, and y
        else:
            # Default scoring (R^2 for regressors)
            return self.model.score(X, y)

    def grid_search(self, X, y, param_grid, cv=5, scoring=None, refit=None):
       grid_search = GridSearchCV(
            estimator=self.model, param_grid=param_grid, cv=cv, 
            scoring=scoring if scoring else 'accuracy', refit=refit, n_jobs=-1, verbose=0
        )
        grid_search.fit(X, y)

        # Store the best model and its parameters
        self.best_params_ = grid_search.best_params_
        self.model = grid_search.best_estimator_  # Update model to the best one
        self.feature_importances_ = self.model.feature_importances_

       st.write( print(f"Best parameters: {self.best_params_}"))
        st.write(print(f"Best score: {grid_search.best_score_}"))
      cv_results = grid_search.cv_results_
        scoring_methods = [key for key in cv_results if key.startswith("mean_test_")]
        
        # Get the index of the best model
        best_index = grid_search.best_index_
        
        # Retrieve the best scores for each scorer
        best_scores = {scorer: cv_results[scorer][best_index] for scorer in scoring_methods}
        
        st.write(print("Best scores for each scorer:"))
for scorer, score in best_scores.items():
           st.write( print(f"{scorer.replace('mean_test_', '')}: {score}"))
        
        return self

    def summarize_results(self, metric="roc_auc", top_n=5):
       if self.cv_results_ is None:
            st.write(print("No grid search results found. Run grid_search first.")
            return None)
    
        # Convert cv_results_ to a DataFrame
        results_df = pd.df(self.cv_results_)
    
        # Select mean and std columns for the chosen metric
        metric_cols = [f"mean_test_{metric}", f"std_test_{metric}"]
        param_cols = [col for col in results_df.columns if col.startswith("param_")]
         results_df = results_df[param_cols + metric_cols]
        results_df = results_df.sort_values(by=f"mean_test_{metric}", ascending=False)
    
        # Return the top N results
        top_results = results_df.head(top_n)
        st.write(print(f"Top {top_n} results for metric: {metric}")
        return top_results)
grid_params_cb = {'depth': [4], 'iterations': [200], 'l2_leaf_reg': [3], 'learning_rate': [0.1], 'logging_level': ['Silent']}
grid_params_xgb = {'colsample_bytree': [0.8], 'learning_rate': [0.1], 'max_depth': [3], 'min_child_weight': [5], 'n_estimators': [200], 'subsample': [1.0]}
grid_params_lgbm = {'colsample_bytree': [0.8], 'learning_rate': [0.05], 'max_depth': [6], 'min_child_samples': [10], 'n_estimators': [100], 'num_leaves': [31], 'subsample': [0.8], 'verbose': [-1]}
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

refit = 'roc_auc'
cv = 5
verbose = 1
n_jobs = -1
cb = ModelWrapper(model_class=CatBoostClassifier)
xgb = ModelWrapper(model_class=XGBClassifier)
lgbm = ModelWrapper(model_class=LGBMClassifier)

st.write(print('CatBoost Classifier Results'))
cb.grid_search(X_train, y_train, param_grid=grid_params_cb, cv=cv, scoring=scoring, refit=refit)
st.write(print())
st.write(print('XGBoost Classifier Results'))
xgb.grid_search(X_train, y_train, param_grid=grid_params_xgb, cv=cv, scoring=scoring, refit=refit)
st.write(print())
st.write(print('LightGBM Classifier Results'))
lgbm.grid_search(X_train, y_train, param_grid=grid_params_lgbm, cv=cv, scoring=scoring, refit=refit)
catboost_features = cb.feature_importances_
xgboost_features = xgb.feature_importances_
lightgbm_features = lgbm.feature_importances_
cols = X_train.columns.values
# Create a dataframe with features
feature_dataframe = pd.df(
    {
        'features': cols,
        'CatBoost Classifier feature importances': catboost_features,
        'XGBoost Classifier feature importances': xgboost_features,
        'LightGBM Classifier feature importances': lightgbm_features
    }
)

st.write(feature_dataframe.head(10))
def plot_feature_importance(model_name):
    # Extract data
    y_values = feature_dataframe[model_name].values
    x_values = np.arange(len(y_values))  # Numerical index for the features
    feature_labels = feature_dataframe['features'].values

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x=x_values,
        y=y_values,
        c=y_values,  # Color the points by their importance values
        cmap='viridis',
        s=100,  # Size of points
        edgecolor='k',
        alpha=0.7
    )
   plt.colorbar(scatter, label="Feature Importance")  # Add color bar

    # Add text labels to the x-axis
    plt.xticks(x_values, feature_labels, rotation=75, ha='right')
    
    # Add labels and title
    plt.xlabel("Features")
    plt.ylabel("Feature Importance")
    plt.title(f"{model_name} Feature Importances", fontsize=14)

    # Adjust layout and show the plot
    plt.tight_layout()
    st.write(plt.show())
    st.write(plot_feature_importance('CatBoost Classifier feature importances')
plot_feature_importance('XGBoost Classifier feature importances')
plot_feature_importance('LightGBM Classifier feature importances'))
#Prediction
if st.button("Prediction"):
	st.subheader("Prediction")
  top10_catboost = feature_dataframe.sort_values(
    by='CatBoost Classifier feature importances', ascending=False
).head(10)

# Sort by XGBoost feature importances and get the top 10
top10_xgboost = feature_dataframe.sort_values(
    by='XGBoost Classifier feature importances', ascending=False
).head(10)

# Sort by LightGBM feature importances and get the top 10
top10_lightgbm = feature_dataframe.sort_values(
    by='LightGBM Classifier feature importances', ascending=False
).head(10)
  st.write(print("Top 10 features for CatBoost:"))
  st.write(print(top10_catboost[['features', 'CatBoost Classifier feature importances']]))

st.write(print("\nTop 10 features for XGBoost:"))
st.write(print(top10_xgboost[['features', 'XGBoost Classifier feature importances']]))

st.write(print("\nTop 10 features for LightGBM:"))
st.write(print(top10_lightgbm[['features', 'LightGBM Classifier feature importances']]))
    
