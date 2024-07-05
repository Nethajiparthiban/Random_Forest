from sklearn.datasets import load_digits
import pandas as pd
dataset=load_digits()
print(dir(dataset))
import matplotlib.pyplot as plt
for i in range(5):
    plt.matshow(dataset.images[i])
    plt.show()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['target']=dataset.target
df['target_names']=df.target.apply(lambda x: dataset.target_names[x])
#print(df.head())
#lets split the data using train split method
x=df.drop(['target','target_names'],axis='columns')
y=df.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# print(len(x_test))
# print(len(x_train))
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=10)
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
from sklearn.metrics import confusion_matrix
y_pred=clf.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True)
plt.show()
import pickle
with open('forest1_pic','wb') as f:
    pickle.dump(clf,f)
with open('forest1_pic','rb') as k:
    model=pickle.load(k)

