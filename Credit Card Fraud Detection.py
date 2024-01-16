'''
Project: Credit card fraud detection
Author: Sai Nayana Sahishna Kandalam
Description: This Program is designed to detect the
            fraudulent transactions with credit card.
Revisions:
	00 -to identify the fraudulent transactions and accuracy of algorithm.
'''
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# for highlevel visualizations, used seaborn library over matplotlib
import seaborn as sns

#for displaying all the plots directly in the notebook interface, as inline graphics.
get_ipython().run_line_magic('matplotlib', 'inline')
# to display all columns of a DataFrame
pd.set_option('display.max_columns', None)

#MODEL SELECTIONS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#Thresholds
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Read data
data=pd.read_csv('creditcard.csv')
data.head()

# Display the data
data.info()

# Get count of distinct data in "Class" column
data['Class'].value_counts()


# Plot histogram of "Class" column
sns.histplot(data['Class'])
plt.yscale('log')
plt.show()


# Grouping data by class and get sum of amount in each class
data.groupby('Class')['Amount'].sum()



x_dummy=data.drop(columns='Class', axis=1)
y=data['Class']


# In[8]:


scaler=StandardScaler()
x=scaler.fit_transform(x_dummy)


# In[9]:


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=123)
print(f'x_train{x_train.shape}\n, x_test{x_test.shape}\n, y_train{y_train.shape}\n, y_test{y_test.shape}')


# In[10]:


def logic_regression(x_train, y_train, x_test):
  lr=LogisticRegression()
  lr.fit(x_train, y_train)
  y_train_pred=lr.predict(x_train)
  y_train_cl_report=classification_report(y_train, y_train_pred, target_names = ['No Fraud', 'Fraud'])
  print("_"*100)
  print("TRAIN MODEL CLASSIFICATION REPORT")
  print("_"*100)
  print(y_train_cl_report)
  y_test_pred=lr.predict(x_test)
  y_test_cl_report=classification_report(y_test, y_test_pred, target_names = ['No Fraud', 'Fraud'])
  print("_"*100)
  print("TEST MODEL CLASSIFICATION REPORT")
  print("_"*100)
  print(y_test_cl_report)
  print("_"*100)
  return y_test_pred, lr



# In[11]:


y_test_pred, lr= logic_regression(x_train, y_train, x_test)



# In[12]:


def conf_mat(y_test, y_test_pred):
  con_mat=confusion_matrix(y_test, y_test_pred)
  labels = ['No Fraud', 'Fraud']
  sns.heatmap(con_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.show()


# In[13]:


conf_mat(y_test, y_test_pred)


# In[14]:


def KNeighbors(x_train, y_train, x_test):
  Kneib=KNeighborsClassifier(n_neighbors=4)
  Kneib.fit(x_train, y_train)
  y_train_pred=Kneib.predict(x_train)
  y_train_cl_report=classification_report(y_train, y_train_pred, target_names = ['No Fraud', 'Fraud'])
  print("_"*50)
  print("TRAIN MODEL CLASSIFICATION REPORT")
  print("_"*50)
  print(y_train_cl_report)
  y_test_pred=Kneib.predict(x_test)
  y_test_cl_report=classification_report(y_test, y_test_pred, target_names = ['No Fraud', 'Fraud'])
  print("_"*50)
  print("TEST MODEL CLASSIFICATION REPORT")
  print("_"*50)
  print(y_test_cl_report)
  print("_"*50)
  return y_test_pred,Kneib


# In[15]:


y_test_pred, Kneib=KNeighbors(x_train, y_train, x_test)


# In[16]:


lr_prob=lr.predict_proba(x_test)
KNeib_prob=Kneib.predict_proba(x_test)
fpr1, tpr1, thresh1=roc_curve(y_test, lr_prob[:,1], pos_label=1)
fpr2, tpr2, thresh2=roc_curve(y_test, KNeib_prob[:,1], pos_label=1)

optimal_thres_lr=thresh1[np.argmax(tpr1 - fpr1)]
optimal_thres_KNeib=thresh2[np.argmax(tpr2 - fpr2)]
# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
print(f" optimal_thres_lr\t {optimal_thres_lr} \n optimal_thres_KNeib\t{optimal_thres_KNeib}")


# In[17]:


opt={'Logistic Regression':optimal_thres_lr,'KNeighbors Classification':optimal_thres_KNeib}
for model, thresh in opt.items():
  if model == 'Logistic Regression':
    y_test_pred_adj=lr.predict_proba(x_test)[:,1]
  elif model =='KNeighbors Classification':
    y_test_pred_adj=Kneib.predict_proba(x_test)[:,1]
    
  y_test_pred_adj1 = (y_test_pred_adj >= thresh).astype(int)
  ac_score = accuracy_score(y_test, y_test_pred_adj1)
  ROC_AC=roc_auc_score(y_test, y_test_pred_adj1)
  
  print("_" * 50)
  print(f"Model: {model}")
  print(f"Threshold: {thresh}")
  print(f"Accuracy Score: {ac_score}")
  print(f"ROC Accuracy Score: {ROC_AC}")
  print("_" * 50)
    
  y_test_cl_report_adj = classification_report(y_test, y_test_pred_adj1, target_names=['No Fraud', 'Fraud'])
  print("_" * 50)
  print("Classification Report:")
  print(y_test_cl_report_adj)
  print("_" * 50)


# In[18]:


# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(fpr2, tpr2, linestyle='-',color='green', label='KNN')
plt.plot(p_fpr, p_tpr, linestyle='dashdot',color='blue', label='RANDOM')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


# In[19]:


data['Class'].value_counts()


# In[20]:


df_0 = data[data['Class'] == 0].sample(n=492, random_state=42)
df_1= data[data['Class'] == 1].sample(n=492, random_state=42)
print(f' Fraud Shape{df_1.shape}\n No Fraud shape{df_0.shape}')


# In[21]:


df_concat=pd.concat([df_0,df_1], ignore_index=True)


# In[22]:


df_concat.shape


# In[23]:


x_bal_dummy=df_concat.drop('Class', axis=1)
y_bal=df_concat['Class']
print(x_bal_dummy.shape, '\n', y_bal.shape)


# In[24]:


x_bal=scaler.fit_transform(x_bal_dummy)


# In[25]:


x_train_b, x_test_b, y_train_b, y_test_b=train_test_split(x_bal,y_bal, test_size=0.20, random_state=123)
print(f'x_train{x_train_b.shape}\n, x_test{x_test_b.shape}\n, y_train{y_train_b.shape}\n, y_test{y_test_b.shape}')


# In[26]:


bal_lr=LogisticRegression()
bal_lr.fit(x_train_b,y_train_b)
bal_pred_train=bal_lr.predict(x_train_b)
bal_pred_test=bal_lr.predict(x_test_b)


# In[27]:


bal_cl_report_train=classification_report(y_train_b,bal_pred_train)
print(bal_cl_report_train)
bal_cl_report_test=classification_report(y_test_b,bal_pred_test)
print(bal_cl_report_test)


# In[28]:


conf_mat(y_train_b,bal_pred_train)


# In[29]:


conf_mat(y_test_b,bal_pred_test)


# In[30]:


knn=KNeighborsClassifier()
knn.fit(x_train_b,y_train_b)
knn_bal_pred_train=bal_lr.predict(x_train_b)
knn_bal_pred_test=bal_lr.predict(x_test_b)


# In[31]:


knn_bal_cl_report_train=classification_report(y_train_b,knn_bal_pred_train)
print(knn_bal_cl_report_train)
knn_bal_cl_report_test=classification_report(y_test_b,knn_bal_pred_test)
print(knn_bal_cl_report_test)


# In[32]:


conf_mat(y_train_b,knn_bal_pred_train)
conf_mat(y_test_b,knn_bal_pred_test)


# In[ ]:




