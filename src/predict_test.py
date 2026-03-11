import joblib
import numpy as np

# Load model
model = joblib.load("../models/churn_model.joblib")

# Example input [AccountWeeks, ContractRenewal, DataPlan, DataUsage, CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins]
sample = np.array([[120, 1, 1, 2.5, 1, 250, 120, 80, 10, 12]])

prediction = model.predict(sample)
print("Prediction:", "Churn" if prediction[0] == 1 else "No Churn")
