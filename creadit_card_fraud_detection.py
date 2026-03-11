import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, auc

df = pd.read_csv('creditcard.csv')
print(df.describe())

df_Y = df['Class']
print(df_Y.head())

df_X = df.drop(columns=['Class'])

X_train, X_test, Y_train, Y_test = train_test_split(df_X,df_Y,test_size=0.2,random_state=42)

#Used cross validation & finalized on Random Forest

model = RandomForestClassifier(n_estimators=200,
    class_weight='balanced',
    random_state=42)
model.fit(X_train,Y_train)
print("Final score: ",model.score(X_test,Y_test))

# Since data is imbalanced, meaning, % of fraudulant cases from given dataset is very low, lets comput AUC score for checking model efficiency
# probability of positive class
y_scores = model.predict_proba(X_test)[:,1]

precision, recall, thresholds = precision_recall_curve(Y_test, y_scores)

pr_auc = auc(recall, precision)

print("AUC: ",pr_auc)

