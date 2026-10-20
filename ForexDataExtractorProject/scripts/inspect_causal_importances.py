import joblib
import pandas as pd

m = joblib.load("models/pipeline_universe_buy_calibrated.joblib")
clf = m.estimator if hasattr(m, "estimator") else m.base_estimator
imp = clf.feature_importances_
cols = m.feature_names_in_
df_imp = pd.DataFrame({"Feature": cols, "Importance": imp}).sort_values("Importance", ascending=False)
print("TOP 25 FEATURES IN CAUSAL MODEL (ZERO LOOKAHEAD):")
print(df_imp.head(25).to_string(index=False))
