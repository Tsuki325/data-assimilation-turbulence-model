
import xgboost
import shap
shap.initjs()

# train an XGBoost model
X, y = shap.datasets.communitiesandcrime()
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)
print(shap_values)
print(shap_values.values.shape,shap_values.base_values.shape)
# visualize the first prediction's explanation
shap.plots.beeswarm(shap_values)


