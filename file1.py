CUSTOMER CHURN PREDICTION SYSTEM

#Importing the dataset
import pandas as pd
df=pd.read_csv("C:/Users/User/Desktop/Churn_Modelling.csv")

#Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['Geography']=le.fit_transform(df['Geography'])

#Removing unnecessary columns in data file
df.drop(columns=['RowNumber','Surname'])

#Checking if the distribution of target is balanced or imbalanced
high_df=df[df.Exited==0]#Values=7963 records
low=df[df.Exited==1]#Values=2037 records

#As the data is imbalanced, we're going to overfit the data and create a new dataset
from sklearn.utils import resample
low_df=resample(low,replace=True,n_samples=7963)
new_df=pd.concat([high_df,low_df])#New dataset 

#Seperating the features and target columns
x=new_df.drop(columns=['RowNumber','Surname','Exited'])
y=new_df['Exited']

#Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)

#training and testing various models
#Importing required modules
from sklearn.metrics import confusion_matrix,classification_report

#1.Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)#Model Training
result_1=rf.score(x_test,y_test)*100 #Model performance score
y_pred_1=rf.predict(x_test) #Model prediction
cm_1=confusion_matrix(y_test,y_pred_1) #Check the accuracy of the model
cr_1=classification_report(y_test,y_pred_1) #Check the overall performance of the model
print(result_1)
print(cm_1)
print(cr_1)

#Model results
'''
Result_1=94.57932189200503

Confusion_matrix:
[[2195  200]
 [  59 2324]]

 Classifiction_Report:
              precision    recall  f1-score   support

           0       0.97      0.92      0.94      2395
           1       0.92      0.98      0.95      2383

    accuracy                           0.95      4778
   macro avg       0.95      0.95      0.95      4778
weighted avg       0.95      0.95      0.95      4778
'''

#2.Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)#Model Training
result_2=lr.score(x_test,y_test)*100 #Model performance score
y_pred_2=lr.predict(x_test) #Model prediction
cm_2=confusion_matrix(y_test,y_pred_2) #Check the accuracy of the model
cr_2=classification_report(y_test,y_pred_2) #Check the overall performance of the model
print(result_2)
print(cm_2)
print(cr_2)

#Model results
'''
Result_1=69.67350355797404

Confusion_matrix:
[[1737  662]
 [ 787 1592]]

 Classification_Report:
              precision    recall  f1-score   support

           0       0.69      0.72      0.71      2399
           1       0.71      0.67      0.69      2379

    accuracy                           0.70      4778
   macro avg       0.70      0.70      0.70      4778
weighted avg       0.70      0.70      0.70      4778
'''

#NOTE: THE SCORES, CONFUSION_MATRICES AND CLASSIFICATION_REPORTS MAY CHANGE WITH SUCCESSIVE RUNS BUT WILL MAINTAIN THE VALUE FOR A CERTAIN RANGE
THANK YOU....
