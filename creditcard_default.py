
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("creditcard.csv")
data = np.array(data)

X = data[:, :-1]
y = data[:, -1]

y = y.astype("int")
x = x.astype("int")

# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# fit the model on the whole dataset
model = XGBClassifier(objective='reg:squarederror')
model.fit(X_train, y_train)
# Predicting the Test set results
y_pred = model.predict(X_test)


inputt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(inputt)]

b = model.predict_proba(final)


pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

