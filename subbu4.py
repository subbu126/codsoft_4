import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.tree import DecisionTreeClassifier      
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Load dataset
data = pd.read_csv(r"c:\Users\durga\Downloads\archive (2).zip", encoding='latin-1')
print(data)
##print(data.head())
##print(data.tail())
##print(data.shape)
##print(data.size)
##print(data.describe())
data.drop_duplicates(inplace=True)
print("Column names:", data.columns.tolist())

# Keep only necessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Remove duplicates
data.drop_duplicates(inplace=True)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    return text.strip()

# Apply preprocessing
data['message'] = data['message'].apply(preprocess_text)

# Mapping labels to binary values
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform test data using the same vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)



### Train and Evaluate Multiple Classifiers 

# Naïve Bayes Classifier

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)
y_pred_nb = nb_classifier.predict(X_test_tfidf)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("\n Naïve Bayes Accuracy:", accuracy_nb)
print(classification_report(y_test, y_pred_nb, target_names=['Legitimate SMS', 'Spam SMS']))

# Logistic Regression

lr_classifier = LogisticRegression()
lr_classifier.fit(X_train_tfidf, y_train)
y_pred_lr = lr_classifier.predict(X_test_tfidf)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("\n Logistic Regression Accuracy:", accuracy_lr)
print(classification_report(y_test, y_pred_lr, target_names=['Legitimate SMS', 'Spam SMS']))

# Support Vector Machine (SVM)
svm_classifier = SVC()
svm_classifier.fit(X_train_tfidf, y_train)
y_pred_svm = svm_classifier.predict(X_test_tfidf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("\nSVM Accuracy:", accuracy_svm)
print(classification_report(y_test, y_pred_svm, target_names=['Legitimate SMS', 'Spam SMS']))

# Random Forest 
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)
y_pred_rf = rf_classifier.predict(X_test_tfidf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("\n Random Forest Accuracy:", accuracy_rf)
print(classification_report(y_test, y_pred_rf, target_names=['Legitimate SMS', 'Spam SMS']))

# K-Nearest Neighbors (KNN) 
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_tfidf, y_train)
y_pred_knn = knn_classifier.predict(X_test_tfidf)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("\nKNN Accuracy:", accuracy_knn)
print(classification_report(y_test, y_pred_knn, target_names=['Legitimate SMS', 'Spam SMS']))

# Decision Tree (Newly Added)
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_tfidf, y_train)
y_pred_dt = dt_classifier.predict(X_test_tfidf)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("\nDecision Tree Accuracy:", accuracy_dt)
print(classification_report(y_test, y_pred_dt, target_names=['Legitimate SMS', 'Spam SMS']))

# Progress bar simulation
progress_bar = tqdm(total=100, position=0, leave=True)
for i in range(10, 101, 10):
    progress_bar.update(10)
    progress_bar.set_description(f'Progress: {i}%')

progress_bar.close()

# Display final results
print("\nFinal Model Accuracies:")
print(f"Naïve Bayes: {accuracy_nb:.2f}")
print(f"Logistic Regression: {accuracy_lr:.2f}")
print(f"Support Vector Machine: {accuracy_svm:.2f}")
print(f"Random Forest: {accuracy_rf:.2f}")
print(f"KNN: {accuracy_knn:.2f}")
print(f"Decision Tree: {accuracy_dt:.2f}")



#Convert label column to categorical if not already
data["label"] = data["label"].astype("category")

#Count of ham and spam
label_counts = data["label"].value_counts()

#Create a Figure with 3 Subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

#Histogram with Smooth KDE Curve
sns.histplot(data["label"], bins=2, color="red", edgecolor="black", ax=axes[0], kde=True)
axes[0].set_xlabel("Message Type")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Ham vs. Spam Messages")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Ham", "Spam"])

#Bar Chart (Fixed)
axes[1].bar(["Ham", "Spam"], label_counts.values, color=["blue", "orange"], edgecolor="black")
axes[1].set_xlabel("Message Type")
axes[1].set_ylabel("Count")
axes[1].set_title("Spam vs Ham Count")

#Pie Chart (Fixed)
axes[2].pie(label_counts, labels=["Ham", "Spam"], autopct="%.1f%%", colors=["Green", "Red"])
axes[2].set_title("Spam vs Ham Distribution")

#Adjust layout and show all plots
plt.tight_layout()
plt.show()