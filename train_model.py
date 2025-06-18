import pandas as pd
from anomaly_detection import train_on_clean

clean_df = pd.read_csv("synthetic_ipps_a_employees_clean.csv")
models = train_on_clean(clean_df)
print("✅  Trained & saved:", ", ".join(models))
