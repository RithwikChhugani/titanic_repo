import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
random.seed(0)

# st.write(st.__version__)

st.header('Welcome to our Titanic Streamlit Web App')
st.subheader('''
	The page is divied in two categories:
		1. PowerBI report on Titanic dataset
		2. Data preprocessing and predictions
	''')

options = st.selectbox('Please Select',['PowerBI','Preprocessing & predictions'])

st.write('\n\n')

if options == 'PowerBI':
	st.markdown("""<iframe width="600" height="400" src="https://app.powerbi.com/view?r=eyJrIjoiMjUyNDQ2YTYtYWY2Yy00NWU0LWJmYTMtOGY2YjBhZjI5NTM2IiwidCI6IjZkYjU5OTA5LTYyMjYtNDQ3My05MDYxLWJhZTNjNjRiY2I4NCIsImMiOjEwfQ%3D%3D&pageName=ReportSection" frameborder="0" allowFullScreen="true"></iframe>""",unsafe_allow_html=True)
else:
	st.set_option('deprecation.showfileUploaderEncoding', False)
	uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
	if uploaded_file is not None:
		df = pd.read_csv(uploaded_file)

		st.write(df.head())
		# dropping non essential columns
		df = df.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)
		# dropping null values
		df = df.dropna() 
		# encoding the required columns
		labelencoder = LabelEncoder()
		df['Pclass'] = labelencoder.fit_transform(df['Pclass'])
		df['Survived'] = labelencoder.fit_transform(df['Survived'])
		df['Embarked'] = labelencoder.fit_transform(df['Embarked'])
		df['Sex'] = labelencoder.fit_transform(df['Sex'])

		st.sidebar.header('User Input Parameters')
		st.sidebar.markdown('Please input the values you would like to predict for:')

		def user_input_features():
			pclass = st.sidebar.selectbox('Pclass', [0,1])
			sex = st.sidebar.selectbox('Sex', [0,1])
			age = st.sidebar.slider('Age', 0.42, 31.00, 80.00)
			sibsp = st.sidebar.slider('SibSp', 0, 2, 5)
			parch = st.sidebar.slider('Parch',0,2,6)
			fare = st.sidebar.slider('Fare',0.0,2.0,513.0)
			embarked = st.sidebar.slider('Embarked',0,2,3)
			data = {'pclass': pclass, 'sex': sex,  'age': age, 'sibsp': sibsp,'parch':parch,'fare':fare,'embarked':embarked}
			features = pd.DataFrame(data, index=[0])
			return features

		df1 = user_input_features()

		# print info and description
		st.write(df.info())
		st.write(df.describe())
		st.subheader('User Input parameters')
		st.write(df1)
		# splitting the dataset
		y = df['Survived']
		X = df.iloc[:,1:]
		# split into train test sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
		
		# making model and fitting
		log_reg = LogisticRegression()
		log_reg.fit(X_train,y_train)
		
		prediction = log_reg.predict(df1)
		prediction_proba = log_reg.predict_proba(df1)

		st.subheader('Prediction Probability')
		st.write(prediction_proba)

		# printing accuracy
		# st.write(accuracy_score(y_test,prediction))
