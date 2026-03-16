#dsfvprint
👉 Use Jupyter Notebook or idle

#Practical-1 :-
Aim :- Data Exploration and Visualization
You must install:
pip install pandas
pip install seaborn
pip install matplotlib
pip install scikit-learn

#Question 1: Complete Data Exploration of IRIS Dataset
1.	Importing required libraries
2.	Loading the dataset
3.	Displaying the first and last 5 rows
4.	Checking dataset structure using info()
5.	Displaying summary statistics using describe()
6.	Checking for missing values
7.	Writing conclusions based on findings (null values, datatypes, dataset size)

#Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset('iris')
#Step 2: First & Last 5 Rows
print("First 5 rows")
print(iris.head())
print("Last 5 rows")
print(iris.tail())
#Step 3: Dataset Structure
print("Dataset Info")
print(iris.info())
#Step 4: Summary Statistics
print("Summary Statistics")
print(iris.describe())
#Step 5: Check Missing Values
print("Missing Values")
print(iris.isnull().sum())

#Question 2: 
Using Python, generate the following plots for the IRIS dataset:
1.	Histogram for all numerical columns
2.	Boxplot for all numerical features
3.	Countplot for species distribution
4.	Scatter plot for sepal_length vs sepal_width
Write the purpose of each plot and the findings observed.

#A) Histogram
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset('iris')
#A) Histogram
iris.hist(figsize=(10,8))
plt.show()
#B) Boxplot
sns.boxplot(data=iris)
plt.show()
#C) Countplot
sns.countplot(x='species', data=iris)
plt.show()
#D) Scatter Plot
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
plt.show()

#Question 3: 
Write a Python program to draw a pairplot of the IRIS dataset using hue='species'.
Based on the plot, answer:
a) Which species is most easily separable?
b) Which features provide the best class separation?
c) Why do sepal measurements show more overlap?

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset('iris')
sns.pairplot(iris, hue='species')
plt.show()


#Question 4: 
Using Python, plot a Scatter Multiple graph where:
•	X-axis = Petal Length
•	Y-axis = Sepal Length, Sepal Width, Petal Width
Explain:
a) The trends observed between Petal Length and each of the 3 attributes
b) Which relationship is strongest and why?

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset('iris')

plt.scatter(iris['petal_length'], iris['sepal_length'], label="Sepal Length")
plt.scatter(iris['petal_length'], iris['sepal_width'], label="Sepal Width")
plt.scatter(iris['petal_length'], iris['petal_width'], label="Petal Width")

plt.xlabel("Petal Length")
plt.ylabel("Values")
plt.legend()
plt.show()

#Question 5:
Write a Python program to generate a Scatter Matrix for all numerical features of the IRIS dataset.
Answer:
a) Which pair of features shows strong correlation?
b) Which feature shows minimum correlation?
c) How are clusters represented in the scatter matrix?
d) Why are petal features more discriminative?

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset('iris')
pd.plotting.scatter_matrix(iris.iloc[:,0:4], figsize=(10,8))
plt.show()

#Question 6: 
Using Python, draw a Parallel Coordinates Plot for the IRIS dataset.
Explain:
a)How parallel coordinates represent multidimensional data
b) How species are visually separated across axes
c) Why petal measurements separate species better
d) How lines indicate clustering 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
iris = sns.load_dataset('iris')
parallel_coordinates(iris, 'species')
plt.show()

#Question 7:
Write a Python program to create a Deviation Chart showing deviation of all features from their mean.
Explain:
a) Which feature fluctuates the most?
b) Which feature is most stable?
c) Why are deviation charts useful?
d) Are there any outliers?

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset('iris')
iris_mean = iris.iloc[:,0:4].mean()
deviation = iris.iloc[:,0:4] - iris_mean
deviation.plot(figsize=(10,6))
plt.show()

#Question 8: 
Plot Andrews Curves for the IRIS dataset.
Explain:
a) How Andrews curves transform rows into continuous curves
b) Why similar shapes indicate similar data points
c) How species clusters appear in Andrews curves
d) Which species shows the largest amplitude and why?

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves
iris = sns.load_dataset('iris')
andrews_curves(iris, 'species')
plt.show()



#PRACTICAL 2 – CLASSIFICATION USING DECISION TREE
You must install:
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib

#Question 1: 
#QUESTION 1:Create decision tree model on iris dataset and predict the Species.                                                                                  

#Step 1 — Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
#Step 2 — Load the Dataset
iris = load_iris()
X = iris.data
y = iris.target
#Step 3 — Split the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
#Step 4 — Create Decision Tree Model
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3
)
#Step 5 — Train the Model
dt_model.fit(X_train, y_train)
#Step 6 — Make Predictions
y_pred = dt_model.predict(X_test)
print(y_pred)
#Step 7 — Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
#Step 8 — Visualize Decision Tree
plt.figure(figsize=(12,8))
plot_tree(
    dt_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.show()



#QUESTION 2: Create decision tree model on mtcars dataset.
#Step 1 — Import Libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
#Step 2 — Load the mtcars Dataset
mtcars = pd.read_csv(r"D:\DSF\Practical\mtcars.csv")
mtcars.head()
#Step 3 — Define Features and Target Variable
mtcars = mtcars.drop(columns=['model'])
X = mtcars.drop('am', axis=1)
y = mtcars['am']
#Step 4 — Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
#Step 5 — Create Decision Tree Model
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3
)
#Step 6 — Train Model
dt_model.fit(X_train, y_train)
#Step 7 — Predict Test Data
y_pred = dt_model.predict(X_test)
#Step 8 — Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
#Step 9 — Visualize Decision Tree
plt.figure(figsize=(14,8))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["Automatic", "Manual"],
    filled=True
)
plt.show()



#Practical-3 :- K-Means & DBSCAN Clustering
What To Install?
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib

#QUESTION 1 :- K-Means Clustering on Iris Dataset
#Step 1 — Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
#Step 2 — Load the Iris Dataset
iris = load_iris()
X = iris.data
#Step 3 — Decide Number of Clusters (K)
k = 3
#Step 4 — Apply K-Means Algorithm
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
#Step 5 — Obtain Cluster Labels and Centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print(labels)
print(centroids)
#Step 6 — Visualize the Clusters
plt.scatter(X[:,2], X[:,3], c=labels)
plt.scatter(centroids[:,2], centroids[:,3], marker='X')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-Means Clustering on Iris Dataset")
plt.show()

#QUESTION 2 :-DBSCAN Using Iris Dataset 
#Step 1 — Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
#Step 2 — Load Dataset
iris = load_iris()
X = iris.data
#Step 3 — Perform Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#Step 4 — Create DBSCAN Model
dbscan = DBSCAN(eps=0.6, min_samples=5)
#Step 5 — Apply DBSCAN Algorithm
labels = dbscan.fit_predict(X_scaled)
#Step 6 — Analyze Clusters
print(np.unique(labels))
#Step 7 — Visualize Clusters
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("DBSCAN Clustering on Iris Dataset")
plt.show()



#PRACTICAL NO. 04 – REGRESSION ANALYSIS
What To Install?
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install statsmodels
pip install scipy
    (OR)
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy

PART 1: Simple Linear Regression (Height → Weight)
#Step 1 – Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
#Step 2 – Load Dataset
height = np.array([151,174,138,186,128,136,179,163,152])
weight = np.array([63,81,56,91,47,57,76,72,62])
df = pd.DataFrame({"height":height,"weight":weight})
print(df)
#Step 3 – Data Visualization
sns.scatterplot(x="height",y="weight",data=df)
sns.regplot(x="height",y="weight",data=df,ci=None,scatter=False)
plt.title("Height vs Weight")
plt.show()
#Step 4 – Check Outliers
sns.boxplot(y=df["weight"])
plt.show()
#Step 5 – Build Regression Model
X = df[["height"]]
y = df["weight"]
lr = LinearRegression()
lr.fit(X,y)
print("Intercept:",lr.intercept_)
print("Slope:",lr.coef_[0])
#Step 6 – OLS Model Summary
X_sm = sm.add_constant(X)
model = sm.OLS(y,X_sm).fit()
print(model.summary())
#Step 7 – Prediction
new_height = pd.DataFrame({"height":[140]})
pred = lr.predict(new_height)
print("Predicted weight:",pred[0])
#Step 8 – Residual Analysis
df["predicted"] = lr.predict(X)
df["residuals"] = df["weight"] - df["predicted"]
plt.scatter(df["predicted"],df["residuals"])
plt.axhline(0)
plt.show()
#Step 9 – Q-Q Plot
sm.qqplot(df["residuals"],line="45")
plt.show()
#Step 10 – Histogram
sns.histplot(df["residuals"],kde=True)
plt.show()

PART 2: Linear Regression (Home Size → Selling Price)
#Step 1 – Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
#Step 2 – Load Data
home_size=np.array([1400,1300,1200,950,900,1000,1300,850,1100])
selling_price=np.array([70,62,65,45,40,53,68,40,55])
df=pd.DataFrame({"home_size":home_size,"selling_price":selling_price})
#Step 3 – Visualization
sns.scatterplot(x="home_size",y="selling_price",data=df)
sns.regplot(x="home_size",y="selling_price",data=df)
plt.show()
#Step 4 – Outliers
sns.boxplot(y=df["selling_price"])
plt.show()
#Step 5 – Model Building
X=df[["home_size"]]
y=df["selling_price"]
lr=LinearRegression()
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)
#Step 6 – Prediction
new_size=pd.DataFrame({"home_size":[1500]})
pred=lr.predict(new_size)
print(pred)

PART 3: Linear Regression (Years of Service → Income)
#Step 1 – Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
#Step 2 – Load Data
years=np.array([11,7,9,5,8,6,10])
income=np.array([17,15,13,12,16,14,18])
df=pd.DataFrame({"years":years,"income":income})
#Step 2 – Visualization
sns.scatterplot(x="years",y="income",data=df)
sns.regplot(x="years",y="income",data=df)
plt.show()
#Step 3 – Build Model
X=df[["years"]]
y=df["income"]
lr=LinearRegression()
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)
#Step 4 – Residual Analysis
df["predicted"]=lr.predict(X)
df["residuals"]=df["income"]-df["predicted"]
#Step 5 – Q-Q Plot
sm.qqplot(df["residuals"],line="45")
plt.show()
#Step 6 – Histogram
sns.histplot(df["residuals"],kde=True)
plt.show()



#PRACTICAL NO. 05 – Association Rule Mining
What To Install?
pip install pandas mlxtend
QUESTION 1: Association Rule Mining using Apriori Algorithm
#Step 1 – Import Libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
#Step 2 – Create Transaction Dataset
transactions = [
['Bread','Milk'],
['Bread','Diaper','Beer','Eggs'],
['Milk','Diaper','Beer','Cola'],
['Bread','Milk','Diaper','Beer'],
['Bread','Milk','Diaper','Cola']
]
#Step 3 – Convert Transactions to Binary Format
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)
#Step 4 – Generate Frequent Itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print(frequent_itemsets)
#Step 5 – Generate Association Rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)
print(rules[['antecedents','consequents','support','confidence','lift']])


QUESTION 2:Association Rule Mining using FP-Growth Algorithm
#Step 1 – Import Libraries
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
#Step 2 – Create Dataset
transactions = [
['Mobile','Charger'],
['Mobile','Earphones','Power Bank'],
['Laptop','Mouse'],
['Mobile','Charger','Earphones'],
['Laptop','Mouse','Keyboard']
]
#Step 3 – Convert to Binary Format
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)
#Step 4 – Generate Frequent Itemsets
frequent_itemsets = fpgrowth(df, min_support=0.4, use_colnames=True)
print(frequent_itemsets)
#Step 5 – Generate Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print(rules[['antecedents','consequents','support','confidence','lift']])

Question 3: Apriori – Retail Store Dataset
#Step 1 – Import Libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
#Step 2 – Dataset
transactions = [
['Rice','Wheat','Oil'],
['Rice','Oil'],
['Wheat','Sugar'],
['Rice','Wheat','Sugar'],
['Rice','Oil','Sugar']
]
# Step 3 – Transaction Encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)
#Step 4 – Frequent Itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print(frequent_itemsets)
#Step 5 – Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print(rules[['antecedents','consequents','support','confidence','lift']])

Question 4: Apriori – Retail Store Dataset
#Step 1 – Import Libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
#Step 2 – Dataset
transactions = [
['Paracetamol','Cough Syrup'],
['Paracetamol','Vitamin C'],
['Paracetamol','Cough Syrup','Vitamin C'],
['Vitamin C'],
['Paracetamol','Vitamin C']
]
#Step 3 – Convert Dataset
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions),
columns=te.columns_)
#Step 4 – FP-Growth
frequent_itemsets = fpgrowth(df, min_support=0.4, use_colnames=True)
print(frequent_itemsets)
#Step 5 – Generate Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print(rules[['antecedents','consequents','support','confidence','lift']])

Question 5: Apriori – Course Recommendation
#Step 1 – Import Libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
#step 2- Dataset
transactions = [
['Python','Data Science'],
['Python','Machine Learning'],
['Data Science','Machine Learning'],
['Python','Data Science','Machine Learning'],
['Python','Data Science']
]
#Step 3-Run Apriori
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions),
columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
print(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules[['antecedents','consequents','support','confidence','lift']])

Question 6: FP-Growth – Bank Services
#Step 1 – Import Libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
#step 2 – Dataset
transactions = [
['ATM','Debit Card'],
['ATM','Debit Card','Internet Banking'],
['Credit Card','Internet Banking'],
['ATM','Credit Card'],
['ATM','Debit Card','Credit Card']
]
#Step 3-Run FP-Growth
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions),
columns=te.columns_)
frequent_itemsets = fpgrowth(df, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print(rules[['antecedents','consequents','support','confidence','lift']])

Question 7: Apriori – Hospital Symptoms
#Step 1 – Import Libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
#step 2 – Dataset
transactions = [
['Fever','Cough'],
['Fever','Headache'],
['Fever','Cough','Headache'],
['Cough'],
['Fever','Cough']
]
#Step 3-Run Apriori
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions),
columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules[['antecedents','consequents','support','confidence','lift']])



#PRACTICAL NO. 06 – Deep Learning: Image Classification with CNNs
what to install?
pip install tensorflow matplotlib

Question:- Build a CNN model for image classification using the MNIST dataset and perform the following:
a) Load and preprocess the dataset.
b) Construct a CNN architecture using convolution and pooling layers.
c) Compile and train the model.
d) Evaluate model performance using test data.
e) Predict the class of a sample image.

# Step 1 – Import Required Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# Step 2 – Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)
print(x_test.shape)
# Step 3 – Data Preprocessing
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
# Step 4 – Build CNN Model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
# Step 5 – Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Step 6 – Train the Model
model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)
# Step 7 – Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)
# Step 8 – Predict Sample Image
plt.imshow(x_test[0].reshape(28,28), cmap='gray')
plt.title("Test Image")
plt.show()
prediction = model.predict(x_test[0].reshape(1,28,28,1))
print("Predicted Digit:", prediction.argmax())


#PRACTICAL NO. 07 – To implement a recommendation system using collaborative filtering technique in Python.
What To Install?
pip install pandas
pip install scikit-learn

Question 1: Online Shopping Recommendation(Item-Based) 
# Step 1 – Import Libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Step 2 – Create Dataset
data = {
'User': ['U1','U1','U2','U2','U3','U3'],
'Item': ['Mobile','Earphones','Mobile','Powerbank','Earphones','Powerbank'],
'Rating': [5,4,4,5,3,4]
}
df = pd.DataFrame(data)
print(df)
# Step 3 – Create User-Item Matrix
matrix = df.pivot_table(index='User', columns='Item', values='Rating').fillna(0)
print(matrix)
# Step 4 – Compute Cosine Similarity
similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(
similarity,
index=matrix.columns,
columns=matrix.columns)
print(similarity_df)
# Step 5 – Recommendation Function
def recommend(item):
    return similarity_df[item].sort_values(ascending=False)
# Step 6 – Recommend Items
print(recommend('Mobile'))

Question 2:Movie Recommendation System 
# Step 1 – Import Libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# step 2- Dataset
data = {
'User': ['U1','U1','U2','U2','U3','U3'],
'Movie': ['Avengers','Titanic','Avengers','Inception','Titanic','Inception'],
'Rating': [5,4,4,5,3,4]
}
df = pd.DataFrame(data)
# Step 3 – User-Movie Matrix
matrix = df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)
print(matrix)
# Step 4 – Similarity
similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(similarity,
index=matrix.columns,
columns=matrix.columns)
print(similarity_df)
# Step 4 – Recommendation
def recommend(item):
    return similarity_df[item].sort_values(ascending=False)
print(recommend('Avengers'))

Question 3:Music Recommendation
# Step 1 – Import Libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Step 2 – Dataset
data = {
'User': ['U1','U1','U2','U2','U3','U3'],
'Song': ['SongA','SongB','SongA','SongC','SongB','SongC'],
'Rating': [5,4,4,5,3,4]
}
df = pd.DataFrame(data)
# Step 3 – Matrix
matrix = df.pivot_table(index='User', columns='Song', values='Rating').fillna(0)
print(matrix)
# Step 4 – Similarity
similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(similarity,
index=matrix.columns,
columns=matrix.columns)
# Step 5 – Recommendation
def recommend(item):
    return similarity_df[item].sort_values(ascending=False)
print(recommend('SongA'))

Question 4:Online Course Recommendation
# Step 1 – Import Libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Step 1 – Dataset
data = {
'Learner': ['L1','L1','L2','L2','L3','L3'],
'Course': ['Python','SQL','Python','ML','SQL','ML'],
'Rating': [5,4,4,5,3,4]
}
df = pd.DataFrame(data)
# Step 2 – Matrix
matrix = df.pivot_table(index='Learner', columns='Course', values='Rating').fillna(0)
print(matrix)
# Step 3 – Similarity
similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(similarity,
index=matrix.columns,
columns=matrix.columns)
# step 4 – Recommendation
def recommend(item):
    return similarity_df[item].sort_values(ascending=False)
print(recommend('Python'))

Question 5:Restaurant Recommendation
# Step 1 – Import Libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Step 2 – Load the Data
data = {
'User': ['L1','L1','L2','L2','L3','L3'],
'Restaurant': ['R1','R2','R1','R3','R2','R3'],
'Rating': [5,4,4,5,3,4]
}
df = pd.DataFrame(data)
# Step 3 – Create the Matrix
matrix = df.pivot_table(index='User', columns='Restaurant', values='Rating').fillna(0)
print(matrix)
# Step 4 – Calculate Similarity
similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(similarity,
index=matrix.columns,
columns=matrix.columns)
# Step 5 – Recommendation Function
def recommend(item):
    return similarity_df[item].sort_values(ascending=False)
print(recommend('R1'))

Question 6:Book Recommendation System
# Step 1 – Import Libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Step 2 – Load the Data
data = {
'User': ['U1','U1','U2','U2','U3','U3'],
'Book': ['BookA','BookB','BookA','BookC','BookB','BookC'],
'Rating': [5,4,4,5,3,4]
}
df = pd.DataFrame(data)
# Step 3 – Create User-Item Matrix
matrix = df.pivot_table(index='User', columns='Book', values='Rating').fillna(0)
print(matrix)
# Step 4 – Calculate Cosine Similarity
similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(similarity,
index=matrix.columns,
columns=matrix.columns)
# step 5- Recommendation
def recommend(item):
    return similarity_df[item].sort_values(ascending=False)
print(recommend('BookA'))



#PRACTICAL NO. 08 – Time Series Forecasting (ARIMA Model)(Air Passenger Data)
What To Install?
pip install pandas
pip install numpy
pip install matplotlib
pip install statsmodels
   (OR)
pip install pandas numpy matplotlib statsmodels

# Step 1 – Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Step 2 – Load the Dataset
data = pd.read_csv("AirPassengers.csv")
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
print(data.head())
# Step 3 – Visualize Time Series
plt.figure()
plt.plot(data['Passengers'])
plt.title('Monthly Air Passengers')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.show()
# Step 4 – Check Stationarity (ADF Test)
result = adfuller(data['Passengers'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])
# Step 5 – Make Series Stationary (Differencing)
data_diff = data['Passengers'].diff().dropna()
plt.figure()
plt.plot(data_diff)
plt.title('First Differenced Series')
plt.show()
# Step 6 – Identify ARIMA Parameters (p, d, q)
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data_diff, lags=20)
plt.show()
# Step 7 – Build ARIMA Model
model = ARIMA(data['Passengers'], order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())
# Step 8 – Forecast Future Values
forecast = model_fit.forecast(steps=12)
print(forecast)
# Step 9 – Plot Actual vs Forecast
plt.figure()
plt.plot(data['Passengers'], label='Actual')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.show()



#PRACTICAL NO. 09 – Anomaly Detection
What To Install?
pip install pandas
pip install scipy
pip install scikit-learn
      (OR)
pip install pandas scipy scikit-learn

Question 1
Detect Salary Outliers
PART 1: Z-SCORE METHOD
# Step 1 – Import Libraries
import pandas as pd
from scipy.stats import zscore
# Step 2 – Load the Dataset
df = pd.DataFrame({
'Employee_ID': ['E01','E02','E03','E04','E05','E06','E07'],
'Salary': [48000,52000,50500,49200,51000,120000,18000]
})
# Step 3 – Calculate Z-scores
df['Z_Score'] = zscore(df['Salary'])
print(df['Z_Score'])
# Step 4 – Identify Outliers
outliers = df[abs(df['Z_Score']) > 2]
print(outliers)


PART 2: IQR METHOD
# Step 1 – Import Libraries
import pandas as pd
from scipy.stats import zscore
# Step 2 – Load the Dataset
data = {
'Employee_ID':['E01','E02','E03','E04','E05','E06','E07'],
'Salary':[48000,52000,50500,49200,51000,120000,18000]
}
df = pd.DataFrame(data)
# Step 3 – Calculate Quartiles
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
# Step 3 – Define Bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Step 4 – Identify Outliers
outliers = df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]
print(outliers)

Question 2:Fraud Transaction Detection (Isolation Forest)
PART 3: ISOLATION FOREST
from sklearn.ensemble import IsolationForest
import pandas as pd
df = pd.DataFrame({
 'Amount': [2800, 3200, 3500, 2900, 3100, 18000, 50]
})
model = IsolationForest(contamination=0.2, random_state=42)
df['Anomaly'] = model.fit_predict(df)
print(df)


Question 3
Delivery Time Outliers (KNN)
PART 4: K-NN METHOD
from sklearn.neighbors import NearestNeighbors
import pandas as pd
df = pd.DataFrame({
 'Delivery_Time': [46, 48, 50, 47, 49, 95, 10]
})
nbrs = NearestNeighbors(n_neighbors=3)
nbrs.fit(df)
distances, _ = nbrs.kneighbors(df)
df['Avg_Distance'] = distances.mean(axis=1)
threshold = df['Avg_Distance'].mean() + 2*df['Avg_Distance'].std()
outliers = df[df['Avg_Distance'] > threshold]
print(outliers)

Question 4
Customer Behavior Outliers (LOF)
PART 5: LOCAL OUTLIER FACTOR (LOF)
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
df = pd.DataFrame({
 'Spending_Score': [45, 50, 48, 52, 49, 98, 5],
 'Purchase_Frequency': [18, 20, 22, 19, 21, 65, 2]
})
lof = LocalOutlierFactor(n_neighbors=3)
df['Anomaly'] = lof.fit_predict(df)
outliers = df[df['Anomaly'] == -1]
print(outliers)

Question 5
Student Marks Outliers
Dataset shown on page 3.
Use same Z-Score and IQR steps.
Outliers:
S06 → 98
S07 → 35

Question 6
Order Amount Fraud (Isolation Forest)
Dataset on page 3-4.
from sklearn.ensemble import IsolationForest
import pandas as pd
df = pd.DataFrame({
'Order_Amount':[1200,1450,1300,1500,1400,12000,80]
})
model = IsolationForest(contamination=0.2, random_state=42)
df['Anomaly'] = model.fit_predict(df)
print(df)
Outliers:
12000
80

Question 7
Hospital Waiting Time (KNN)
Dataset on page 4.
Outliers detected:
90 minutes
5 minutes

Question 8
Fitness Activity Outliers (LOF)
Dataset shown on page 4.
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
df = pd.DataFrame({
'Steps_Per_Day':[6500,7000,6800,7200,6900,18000,1200],
'Workout_Sessions':[4,5,6,5,4,15,1]
})
lof = LocalOutlierFactor(n_neighbors=3)
df['Outlier'] = lof.fit_predict(df)
outliers = df[df['Outlier']==-1]
print(outliers)
Outliers:
(18000 steps, 15 workouts)
(1200 steps, 1 workout)



#PRACTICAL NO. 10 – Text Mining
What To Install?
pip install scikit-learn
pip install pandas
pip install numpy
 (OR)
pip install scikit-learn pandas numpy

Question 1:DOCUMENT CLASSIFICATION(Work vs Personal Email)
#Step 1 – Import Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
#Step 2 – Create Dataset
texts = [
"Meeting scheduled with project team",
"Family dinner this weekend",
"Project deadline extended",
"Birthday party invitation"
]
labels = ["Work","Personal","Work","Personal"]
#Step 3 – Convert Text → TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)
#Step 4 – Train Model
model = MultinomialNB()

model.fit(X, labels)
#Step 5 – Prediction
prediction = model.predict(
vectorizer.transform(["Project meeting tomorrow"])
)
print(prediction)

Question 2: SENTIMENT ANALYSIS (Positive vs Negative Reviews)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
reviews = [
 "The product is amazing",
 "Very bad quality",
 "I am happy with the purchase",
 "Worst experience ever"
]
sentiment = ["Positive", "Negative", "Positive", "Negative"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)
model = LogisticRegression()
model.fit(X, sentiment)
print(model.predict(
    vectorizer.transform(["The product quality is good"])
))


Question 3: SEARCH ENGINE (Information Retrieval) (Find Relevant Document)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
documents = [
 "Data science and machine learning",
 "Introduction to text mining",
 "Python for data analysis"
]
query = ["text mining"]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents + query)
similarity = (tfidf * tfidf.T).toarray()
print("Most relevant document index:",
np.argmax(similarity[-1][:-1]))


Question 4: SPAM DETECTION
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
messages = [
 "Win a free lottery now",
 "Meeting scheduled tomorrow",
 "Urgent offer claim now",
 "Project discussion today"
]
labels = ["Spam", "Not Spam", "Spam", "Not Spam"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)
model = MultinomialNB()
model.fit(X, labels)
print(model.predict(
    vectorizer.transform(["Free offer just for you"])
))


#5.RECOMMENDATION USING COSINE SIMILARITY
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

movies = [
 "Romantic comedy with a fun storyline",
 "Action thriller with suspense",
 "Adventure in a magical world"
]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(movies)
similarity = cosine_similarity(tfidf)
print(similarity)


Question 6
News Classification
texts = [
"Football match ends with thrilling goal",
"Government passes new teaching bill",
"Cricket world cup semi-final today",
"Election campaigns begin in several states"
]
labels = ["Match","Gov work","Match","Gov work"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)
print(
model.predict(
vectorizer.transform(["Project meeting tomorrow"])
)
)


Question 7
Customer Feedback Sentiment
Same steps as Question 2.
Prediction example:
Positive

Question 8
Book Search System
Uses TF-IDF similarity.
Result:
Most relevant document index

Question 9
Spam Email Detection
Same process as Question 4.
Output:
Spam

Question 10
Movie Recommendation System
movies = [
"Romantic comedy with a fun storyline",
"Action thriller with suspense",
"Adventure in a magical world"
]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(movies)
similarity = cosine_similarity(tfidf)
print(similarity)
Conclusion:
Movies belong to different genres, so similarity is low.
