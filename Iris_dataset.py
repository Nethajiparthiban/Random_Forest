from sklearn.datasets import load_iris
dataset=load_iris()
print(dir(dataset))
#converting dataset into DataFrame
import pandas as pd
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['target']=dataset.target
df['flower_Names']=df.target.apply(lambda x: dataset.target_names[x])
df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]
import matplotlib.pyplot as plt
# plt.title('setosa')
# plt.xlabel('sepal length (cm)')
# plt.ylabel('sepal width (cm)')
# plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='blue',marker='+')
# plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='green',marker='.')
# plt.show()
#Lets train split our model to train
x=df.drop(['target','flower_Names'],axis="columns")
y=df.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(y_test)
len(x_train)
#lets implement random forest algoritham
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=10)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
clf.score(x_test,y_test)
#lets check our model where it making wrong using confussion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import seaborn as sns
plt.title('Confussion matrix')
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True)
plt.show()
import pickle
with open('iris_pic','wb') as f:
    pickle.dump(clf,f)
with open('iris_pic','rb') as k:
    model=pickle.load(k)
print(model.predict([[5.8,2.8,5.1,2.4]]))