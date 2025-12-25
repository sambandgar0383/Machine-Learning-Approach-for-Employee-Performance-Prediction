# 1️⃣ Import Libraries
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2️⃣ Load Dataset
data = pd.read_csv("employee_data.csv")
print("Dataset Preview:")
print(data.head())

# 3️⃣ Data Visualization
sns.countplot(x="Performance", data=data)
plt.title("Employee Performance Distribution")
plt.savefig("performance_distribution.png")
print("Plot saved as performance_distribution.png")

# 4️⃣ Data Preprocessing
label_encoder = LabelEncoder()
data["Department"] = label_encoder.fit_transform(data["Department"])
data["Performance"] = label_encoder.fit_transform(data["Performance"])

# Check Performance label encoding
print("Performance label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# 5️⃣ Features and Target
X = data.drop("Performance", axis=1)
y = data["Performance"]

# 6️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7️⃣ Model Training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 8️⃣ Prediction on Test Set
y_pred = model.predict(X_test)

# 9️⃣ Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 🔹 Predict New Employee Performance
# Example: Age=29, Experience=5, Department="IT", TrainingHours=30
dept_encoded = label_encoder.transform(["IT"])[0]
new_employee = [[29, 5, dept_encoded, 30]]
prediction = model.predict(new_employee)

performance_labels = list(label_encoder.classes_)  # Get actual labels
print("\nPredicted Performance Level for new employee:", performance_labels[prediction[0]])
