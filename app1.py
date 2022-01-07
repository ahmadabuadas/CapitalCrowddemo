
 
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import s3fs
 
train = pd.read_csv('C:/Users/aabuadas/crowd/Crowd.csv') 
#fs = s3fs.S3FileSystem(anon=False)
#content = read_file("crowdcapital/Crowd.csv")

train['Gender']= train['Gender'].map({'Male':0, 'Female':1})
train['Married']= train['Married'].map({'No':0, 'Yes':1})
train['Loan_Status']= train['Loan_Status'].map({'N':0, 'Y':1})


train = train.dropna()
#train.isnull().sum()

X = train[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = train.Loan_Status

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size = 0.2, random_state = 10)


from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(max_depth=4, random_state = 10) 
model.fit(x_train, y_train)


from sklearn.metrics import accuracy_score
pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)

pred_train = model.predict(x_train)
accuracy_score(y_train,pred_train)

import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()


# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)

#def myplot():
matrix = train.corr()
f1, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu",annot=True) 


totalloans= train["Loan_Status"].value_counts()
lable = ["No","Yes"]
value = [totalloans[0],totalloans[1]]
f2, ax1 = plt.subplots(figsize=(7, 7))
ax1.pie(value,labels=lable, autopct='%1.0f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  
 
 
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
 
    # Pre-processing user input    
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1
 
    if Married == "Unmarried":
        Married = 0
    else:
        Married = 1
 
    if Credit_History == "Unclear Debts":
        Credit_History = 0
    else:
        Credit_History = 1  
 
    LoanAmount = LoanAmount / 1000
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])
     
    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred
     
def bodyinfo():
	
	st.markdown("## KPI First Row")
           
	# kpi 1 

	kpi1, kpi2 = st.columns(2)

	with kpi1:
            st.markdown("**Credit_History & Loan_Status**")
            CreditHistory = pd.crosstab(train["Credit_History"],train["Loan_Status"])
            st.table(CreditHistory)
            CreditHistory.div(CreditHistory.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
            plt.xlabel("Credit_History")
            plt.ylabel("Percentage")
            st.pyplot(plt)       
        
	with kpi2:
            st.markdown("**Education & Loan_Status**")
            Education = pd.crosstab(train["Education"],train["Loan_Status"])
            st.table(CreditHistory)
            CreditHistory.div(CreditHistory.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
            plt.xlabel("Education")
            plt.ylabel("Percentage")
            st.pyplot(plt) 
            
	#st.markdown("<hr/>",unsafe_allow_html=True)


	st.markdown("## KPI Second Row")
	# kpi 1 

	kpi01, kpi02 = st.columns(2)

	with kpi01:
            st.markdown("**Gender & Loan_Status**")
            Gender = pd.crosstab(train["Gender"],train["Loan_Status"])
            st.table(CreditHistory)
            CreditHistory.div(CreditHistory.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
            plt.xlabel("Gender")
            plt.ylabel("Percentage")
            st.pyplot(plt) 
            

	with kpi02:
            st.markdown("**Married & Loan_Status**")
            Married = pd.crosstab(train["Married"],train["Loan_Status"])
            st.table(CreditHistory)
            CreditHistory.div(CreditHistory.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
            plt.xlabel("Married")
            plt.ylabel("Percentage")
            st.pyplot(plt) 
            

	#st.markdown("<hr/>",unsafe_allow_html=True)

	st.markdown("## Chart Layout")
    #st.pyplot(myplot())
	#st.area_chart(st.pyplot(f))
    #st.pyplot(f)
    #st.plotly_chart(f, use_container_width=True)
    #st.bokeh_chart(f)
	kpi01, kpi02 = st.columns([2,2])

	with kpi01:
            st.pyplot(f1) 
            
	with kpi02:
            st.pyplot(f2)     

	   

       

	
        
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:lightgreen;padding:10px"> 
    <h1 style ="color:black;text-align:center;">Crowd Capital Risk Analysis</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Gender = st.sidebar.selectbox('Gender',("Male","Female"))
    Married = st.sidebar.selectbox('Marital Status',("Unmarried","Married")) 
    ApplicantIncome = st.sidebar.number_input("Applicants monthly income") 
    LoanAmount = st.sidebar.number_input("Total loan amount")
    Credit_History = st.sidebar.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.sidebar.button("Predict"): 
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
        st.sidebar.success('recomanded the loan to {} by {}% certan for {} '.format(result,round(accuracy_score(y_cv,pred_cv),3),LoanAmount))
        print(LoanAmount)
     
if __name__=='__main__': 
    main()
    bodyinfo()
    