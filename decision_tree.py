import pandas as pd
from sklearn import tree
from sklearn.tree import plot_tree, export_text
import matplotlib.pyplot as plt

# read CSV
df = pd.read_csv("politicians.csv")

# drop politician_id column
df = df.drop(columns="politician_id")

# define X and y
X = df.drop(columns="privatization")
y = df["privatization"]

# create and fit decision tree
dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(X.values, y.values)

# visualize decision tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(dtree, feature_names = X.columns, class_names=["Yes","No"], filled = True)
plt.show()


# predict privatization for input: gun_control: 1, abortion: 0, welfare_state: 0, age_criminal: 0
print("Prediction for input: gun_control: 0, abortion: 0, welfare_state: 0, age_criminal: 0")
print(dtree.predict([[0,0,0,0]]))
print("Prediction for input: gun_control: 1, abortion: 1, welfare_state: 1, age_criminal: 0")
print(dtree.predict([[1,1,1,0]]))