import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils import resample
df=pd.read_csv("C:/Users/User/Desktop/Churn_Modelling.csv")

le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['Geography']=le.fit_transform(df['Geography'])
df.drop(columns=['RowNumber','Surname'])
high_df=df[df.Exited==0]
low=df[df.Exited==1]

low_df=resample(low,replace=True,n_samples=7963)
new_df=pd.concat([high_df,low_df])

x=new_df.drop(columns=['RowNumber','Surname','Exited'])
y=new_df['Exited']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


rf=RandomForestClassifier()
rf.fit(x_train,y_train)
result_1=rf.score(x_test,y_test)*100
y_pred_1=rf.predict(x_test)
cm_1=confusion_matrix(y_test,y_pred_1)
cr_1=classification_report(y_test,y_pred_1)
print(result_1)
print(cm_1)
print(cr_1)


lr=LogisticRegression()
lr.fit(x_train,y_train)
result_2=lr.score(x_test,y_test)*100
y_pred_2=lr.predict(x_test)
cm_2=confusion_matrix(y_test,y_pred_2)
cr_2=classification_report(y_test,y_pred_2)
print(result_2)
print(cm_2)
print(cr_2)
