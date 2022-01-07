

#Train_data =  pd.read_csv('C:/Users/aabuadas/crowd/train.csv') 


# separate the data into features and target
#features =  np.array(Train_data.iloc[:, 0:12].values)

#target =  np.array(Train_data.iloc[:, 12:13].values)

# split the data into train and test


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import streamlit as st
#import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.filterwarnings("ignore")


Train_data = pd.read_csv("C:/Users/aabuadas/crowd//train.csv")

Train_data["Gender"].fillna(Train_data["Gender"].mode()[0],inplace=True)
Train_data["Married"].fillna(Train_data["Married"].mode()[0],inplace=True)
Train_data['Dependents'].fillna(Train_data["Dependents"].mode()[0],inplace=True)
Train_data["Self_Employed"].fillna(Train_data["Self_Employed"].mode()[0],inplace=True)
Train_data["Credit_History"].fillna(Train_data["Credit_History"].mode()[0],inplace=True)
Train_data["LoanAmount"].fillna(Train_data["LoanAmount"].mode()[0],inplace=True)
Train_data["Loan_Amount_Term"].fillna(Train_data["Loan_Amount_Term"].mode()[0],inplace=True)

Train_data["TotalIncome"]=Train_data["ApplicantIncome"]+Train_data["CoapplicantIncome"]
Train_data["TotalIncome_log"]=np.log(Train_data["TotalIncome"])
#sns.distplot(Train_data["TotalIncome_log"])


Train_data["EMI"]=Train_data["LoanAmount"]/Train_data["Loan_Amount_Term"]
Train_data["Balance_Income"] = Train_data["TotalIncome"]-Train_data["EMI"]*1000

train=Train_data.drop(["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"],axis=1)
X=train.drop("Loan_Status",1)
X=train.drop("Loan_ID",1)

y=train[["Loan_Status"]]
y= y["Loan_Status"].map(dict(Y=1, N=0))

X = pd.get_dummies(X)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=1)

UI_DataFram = {'Credit_History': ["Good", "Bad"], 'TotalIncome': [5000,10000,15000,20000] ,'Gender': ["Male", "Female"],
               'Married': ["Yes", "No"], 'Dependents': [0, 1, 2, 3], 'Education': ["Graduate", "NotGraduate"], 'Self_Employed': ["Yes", "No"],
               'Property_Area': ["Urban", "Rural", "Semiurban"],'LoanAmount':[100000,150000,200000,250000], 'Loan_Amount_Term':[120,180,360],'Income_bin': ["Very_high", "High","Average","Low"]} 
#x_train, x_test, y_train, y_test = train_test_split(
  #  features, target, test_size=0.2, stratify=target
#)


class StreamlitApp:
    
    def get_table(self, a, b):
        datatable= pd.crosstab(Train_data[a],Train_data[b])
        return datatable
        
    def myplot(self):
        matrix = Train_data.corr()
        f, ax = plt.subplots(figsize=(10, 12))
        sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu",annot=True)
        
    def __init__(self):
        self.model = RandomForestClassifier()

    def train_data_f(self):
        self.model.fit(x_train, y_train)
        return self.model

    def construct_sidebar(self):

        #cols = [col for col in UI_DataFram.columns]

        st.sidebar.markdown(
            
            '<p class="header-style">Crowd Capital Risk Managment</p>',
            unsafe_allow_html=True
        )
        Credit_History = st.sidebar.selectbox(
            #f"Select {cols[0]}",
            f"Select Credit_History",
            #sorted(UI_DataFram[cols[0]].unique())
            UI_DataFram['Credit_History']
        )

        TotalIncome = st.sidebar.selectbox(
            #f"Select {cols[1]}",
            f"Select TotalIncome",
            sorted(x_train['TotalIncome'].unique())
            
            #UI_DataFram['TotalIncome']
        )

        Gender = st.sidebar.selectbox(
            #f"Select {cols[5]}",
            f"Select Gender",
            #sorted(UI_DataFram[cols[5]].unique())
            UI_DataFram['Gender']
        )
        
        Married = st.sidebar.selectbox(
            #f"Select {cols[6]}",
            #sorted(UI_DataFram[cols[6]].unique())
            f"Married",
            UI_DataFram['Married']
        )
        Dependents = st.sidebar.selectbox(
            #f"Select {cols[7]}",
            #sorted(UI_DataFram[cols[7]].unique())
            f"Dependents",
             UI_DataFram['Dependents']
        )
        Education = st.sidebar.selectbox(
            #f"Select {cols[8]}",
            #sorted(UI_DataFram[cols[8]].unique())
            f"Education",
            UI_DataFram['Education']
        )
        Self_Employed = st.sidebar.selectbox(
            #f"Select {cols[9]}",
            #sorted(UI_DataFram[cols[9]].unique())
            f"Self_Employed",
            UI_DataFram['Self_Employed']
        )
        
        Property_Area = st.sidebar.selectbox(
            #f"Select {cols[9]}",
            #sorted(UI_DataFram[cols[9]].unique())
            f"Property_Area",
            UI_DataFram['Property_Area']
        )
        
        LoanAmount = st.sidebar.selectbox(
            #f"Select {cols[9]}",
            #sorted(UI_DataFram[cols[9]].unique())
            f"LoanAmount",
            UI_DataFram['LoanAmount']
        )
        
        Loan_Amount_Term = st.sidebar.selectbox( 
            #f"Select {cols[9]}",
            #sorted(UI_DataFram[cols[9]].unique())
            f"Loan_Amount_Term",
            UI_DataFram['Loan_Amount_Term']
        )
        Income_bin = st.sidebar.selectbox( 
            #f"Select {cols[9]}",
            #sorted(UI_DataFram[cols[9]].unique())
            f"Income_bin",
            UI_DataFram['Income_bin']
        )
        
        Credit_History = 1 if 'Good' else 0
        Gender_Female = 1 if 'Female' else 0
        Gender_Male = 1 if 'Male' else 0
        Married_No = 1 if 'No' else 0
        Married_Yes = 1 if 'Yes' else 0
        TotalIncome_log = np.log(TotalIncome)
        EMI = LoanAmount/Loan_Amount_Term
        Balance_Income = TotalIncome - EMI*1000
        Dependents_0 = 1 if 0 else 0
        Dependents_1 = 1 if 1 else 0
        Dependents_2 = 1 if 2 else 0
        Dependents_3 = 1 if 4 else 0
        Education_Graduate = 1 if 'Graduate' else 0
        Education_NotGraduate =  1 if 'NotGraduate' else 0
        Self_Employed_No =1 if 'No' else 0
        Self_Employed_Yes = 1 if 'Yes' else 0
        Property_Area_Rural  = 1 if 'Rural' else 0                           
        Property_Area_Semiurban = 1 if 'Semiurban' else 0
        Property_Area_Urban = 1 if 'Urban' else 0
        Income_bin_Low = 1 if 'Low' else 0                      
        Income_bin_Average = 1 if 'Average' else 0
        Income_bin_High = 1 if 'High' else 0
        Income_bin_Very_high = 1 if 'Very_high' else 0
        Loan_Status_N=0
        Loan_Status_Y= 1
        
        values =[Credit_History,TotalIncome,TotalIncome_log,EMI,Balance_Income,Gender_Female,Gender_Male,Married_No,Married_Yes,Dependents_0,Dependents_1,Dependents_2,Dependents_3,Education_Graduate,Education_NotGraduate,Self_Employed_No,Self_Employed_Yes,Property_Area_Rural,Property_Area_Semiurban,Property_Area_Urban,Loan_Status_N,Loan_Status_Y]
         
       
        values = values[0:23]
        return values


    def construct_app(self):
        
        st.title("Crowd Capital")
        self.train_data_f()
        values = self.construct_sidebar()
         
        values_to_predict = np.array(values).reshape(1, -1)
        result = self.model.predict(values_to_predict)
        #score_RandomForest = accuracy_score(result,y_test)*100 
        #score_RandomForest
        
        
        
        column_1, column_2 = st.beta_columns(2)
        column_1.markdown(
            f'<p class="font-style" >values </p>',
            unsafe_allow_html=True
        )
        column_1.write(f"{values}")
       
        st.write(f"For this new customer information")
        st.write(f"{values}")
        st.write(f"the System recommend to {result} with certain rate{88.8}")
       
        
        datatable1 = self.get_table("Credit_History","Loan_Status")
        st.markdown("relation between Credit_History Loan_Status ")
        st.table(datatable1)# will display the table
        
        CreditHistory = pd.crosstab(Train_data["Credit_History"],Train_data["Loan_Status"])
        CreditHistory.div(CreditHistory.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
        plt.xlabel("Credit_History")
        plt.ylabel("Percentage")
        st.pyplot(plt)
        
        
        datatable2 = self.get_table("Education","Loan_Status")
        st.markdown("relation between Education Loan_Status ")
        st.table(datatable2)# will display the table
       
        Education = pd.crosstab(Train_data["Education"],Train_data["Loan_Status"])
        Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
        plt.xlabel("Education")
        plt.ylabel("Percentage")
        st.pyplot(plt)
        
        datatable3 = self.get_table("Gender","Loan_Status")
        st.markdown("relation between Education Loan_Status ")
        st.table(datatable3)# will display the table
        Gender = pd.crosstab(Train_data["Gender"],Train_data["Loan_Status"])
        Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
        plt.xlabel("Gender")
        plt.ylabel("Percentage")
        st.pyplot(plt)
          
        
        st.pyplot(self.myplot())

        return self


sa = StreamlitApp()
sa.construct_app()