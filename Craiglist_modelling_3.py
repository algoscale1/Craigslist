# Importing all necessary packages and modules

import re
import string

import pandas as pd
import numpy as np
import pickle
import dill
import matplotlib.pyplot as plt

from random import sample
from sklearn.externals import joblib
from collections import Counter


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Imputer
from sklearn.linear_model import SGDClassifier

def load_data():
	# Loading the CSV
	data_read = pd.read_csv("Full_data_3_stemmed.xlsx", encoding="ISO-8859-1", error_bad_lines=False)
	
	return data_read

def preprocesing_data():

	data_read = load_data()
	# Fill Missing Phone values with 0
	data_read["Phone"].fillna(str(0), inplace=True)
	# Fill Missing Name values with 0
	data_read["Name"].fillna(str(0), inplace=True)
	# Replace Unknown values with 0
	data_read['AdToCity(Distance Functionality)'] = data_read['AdToCity(Distance Functionality)'].replace('Unknown' , np.nan)
	# Replace Unknown values with 0
	data_read['AdToPhoneArea(Distance Functionality)'] = data_read['AdToPhoneArea(Distance Functionality)'].replace('Unknown' , np.nan)
	# Replace Unknown values with 0
	data_read['CitytoPhoneArea(Distance Funcionality)'] = data_read['CitytoPhoneArea(Distance Funcionality)'].replace('Unknown' , np.nan)
	# Initializing Imputer to impute missing values in distance data
	imputer = Imputer(missing_values="NaN", strategy="median")
	# Imputing missing values in AdToCity column with median values
	data_read['AdToCity(Distance Functionality)'] = imputer.fit_transform(data_read[['AdToCity(Distance Functionality)']]).ravel()
	# Imputing missing values in AdToPhoneArea column with median values
	data_read['AdToPhoneArea(Distance Functionality)'] = imputer.fit_transform(data_read[['AdToPhoneArea(Distance Functionality)']]).ravel()
	# Imputing missing values in CityToPhoneArea column with median values
	data_read['CitytoPhoneArea(Distance Funcionality)'] = imputer.fit_transform(data_read[['CitytoPhoneArea(Distance Funcionality)']]).ravel()
	
	# Replace null values with 0.0
	data_read['PhoneAverageMarketPrice '] = data_read['PhoneAverageMarketPrice '].replace('null' , 0.0)
	
	# Replace 'Georgia ' values with 'Georgia'
	data_read['State'] = data_read['State'].replace('Georgia ' , 'Georgia')
	# Fill Missing Image values with 0
	data_read["Image Count"].fillna(float(0.0), inplace=True)
	# Drop Missing Content and Email data for tf-idf step ahead
	data_read.dropna(subset=["Content", "E-mail", "Title", "Content Word Count"], inplace=True)

	# Take subset of features to be used in modelling
	data_read = data_read[['Status', 'City', 'State','Title Word Count', 'Content Word Count', 
	'E-mail', 'Content', 'Title', 'Phone', 'Label', 'Image Count',
	'numberOfupdates','Price', 'Name', 'AdToCity(Distance Functionality)',
	'AdToPhoneArea(Distance Functionality)', 'CitytoPhoneArea(Distance Funcionality)',
	'PhoneAverageMarketPrice ', 'Matched Make(Averagepricefunctionality )',
	"special_char_number in body", "personal_email", "external_link"]]
	
	# performing get dummies operation on categorical features
	data_read = pd.concat([data_read, pd.get_dummies(data_read[['Status', 'City', 'State',
		'Matched Make(Averagepricefunctionality )']])], axis=1)
	# Reshuffling the data to get balanced data
	data_read = data_read.sample(frac=1).reset_index(drop=True)
	
	return data_read

def scam_and_not_scam():

	clean_data_read = preprocesing_data()
	# Extract the Scam data from original data
	scam = clean_data_read[clean_data_read["Label"].str.contains("Scam")]
	
	# Extract the not_scam data from original data
	not_scam = clean_data_read[clean_data_read["Label"].str.contains("Trusworthy")]
	
	# Replace not_scam values with not_scam
	not_scam.is_copy = False
	not_scam['Label'] = not_scam['Label'].replace('Trusworthy' , 'not_scam')
	
	return (scam, not_scam, clean_data_read)

def undersampling():

	scam, not_scam, clean_data_read = scam_and_not_scam()
	
	# Count of total scam labels and not_scam labels
	count_dict = dict(Counter(clean_data_read["Label"]))
	
	# Undersampling of not_scam data to get balanced scam and not_scam labels
	not_scam_undersampled = not_scam.take(np.random.permutation(len(not_scam))[:count_dict['Scam']])
	
	# Return scam and undersampled not_scam data
	return scam, not_scam_undersampled

def data_for_modelling():

	scam, not_scam_undersampled = undersampling()

	# Join undersampled not_scam data with scam data
	df = not_scam_undersampled.append(scam, ignore_index=True)
	# Reshuffling the data to get balanced and evenly distributed data
	df = df.sample(frac=1).reset_index(drop=True)
	
	# Return the dataframe
	return df


def special_chars_count_content_func(df):

	# Function to count special characters in Content
	X_list = df["Content"].tolist()

	punctuation_count_list = []
	special_chars_count_list = []
	count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
	
	for content in X_list:
		punct_count = count(content, string.punctuation)
		special_chars_count = sum(bool(re.match(r"""[!.><:;'@#~{}\[\]-_+=£$%^&()?]""", c)) for c in content)
		special_chars_count_list.append(special_chars_count)

	# If count of special characters in content is greater than equal to 6 then 1 else 0
	final = [1 if i >= 6 else 0 for i in special_chars_count_list]
	
	return final

def special_chars_count_title_func(df):

	# Function to count special characters in Title
	X_list = df["Title"].tolist()

	punctuation_count_list = []
	special_chars_count_list = []
	count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
	
	for content in X_list:
		punct_count = count(content, string.punctuation)
		special_chars_count = sum(bool(re.match(r"""[!.><:;'@#~{}\[\]-_+=£$%^&()?]""", c)) for c in content)
		special_chars_count_list.append(special_chars_count)

	# If count of special characters in content is greater than equal to 3 then 1 else 0
	final = [1 if i >= 3 else 0 for i in special_chars_count_list]
	
	return final

def count_spec_chars_title(df):

	# Function to count special characters in Title
	X_list = df["Title"].tolist()

	punctuation_count_list = []
	special_chars_count_list = []
	count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
	
	for content in X_list:
		punct_count = count(content, string.punctuation)
		special_chars_count = sum(bool(re.match(r"""[!.><:;'@#~{}\[\]-_+=£$%^&()?]""", c)) for c in content)
		special_chars_count_list.append(special_chars_count)

	return special_chars_count_list

def Content_word_count(df):

	# Function for Content word count
	X_list = df['Content Word Count'].tolist()
	
	# If count of words in content is greater than equal to 75 then 1 else 0
	final = [1 if i >= 75 else 0 for i in X_list]
	
	return final


def Image_presence(df):

	# Function to check if Image is present or not
	X_list = df['Image Count'].tolist()
	
	# If count of images in content is greater than equal to 3 then 1 else 0
	final = [1 if i > 3 else 0 for i in X_list]
	
	return final

def craigslist_email_presence(df):

	# Function to check if Craiglist email handle is present or not
	X_list = df["E-mail"].tolist()
	email_list = []
	
	# If present then 1 else 0
	for content in X_list:
		match = re.findall(r'[\w\.-]+@sale.craigslist.org', content)
		if len(match) >= 1:
			email_list.append(1)
		else:
			email_list.append(0)

	return email_list

def Phone_presence(df):

	# Function to check if Phone number is present or not
	X_list = df["Phone"].tolist()
	Phone_presence_list = []
	
	# If present then 1 else 0
	for phone in X_list:
		if phone == '0':
			Phone_presence_list.append(0)
		else:
			Phone_presence_list.append(1)

	return Phone_presence_list

def Name_presence(df):

	# Function to check if Name is present or not
	X_list = df["Name"].tolist()
	Name_presence_list = []
	
	# If present then 1 else 0
	for name in X_list:
		if name == '0':
			Name_presence_list.append(0)
		else:
			Name_presence_list.append(1)

	return Name_presence_list


def SpecialCharsInContent(df):

	# Function to check if Special character is present in content or not
	X_list = df['special_char_number in body'].tolist()
	SpecialCharsInContent_list = []
	
	# If value is 0 then 0 else 1
	for value in X_list:
		if value == 0:
			SpecialCharsInContent_list.append(0)
		else:
			SpecialCharsInContent_list.append(1)

	return SpecialCharsInContent_list

def SpecialCharsInTitle(df):

	# Function to check if Special character is present in title or not
	X_list = df['count_spec_chars_title'].tolist()
	SpecialCharsInTitle_list = []
	
	# If value is 0 then 0 else 1
	for value in X_list:
		if value == 0:
			SpecialCharsInTitle_list.append(0)
		else:
			SpecialCharsInTitle_list.append(1)

	return SpecialCharsInTitle_list

def appending_to_dataframe():

	data_read = preprocesing_data()

	# Appending the above extracted features to the undersampled data
	
	df = data_for_modelling()
	df.is_copy = False
	df["special_chars_count_content"] = special_chars_count_content_func(df)
	df["special_chars_count_title"] = special_chars_count_title_func(df)
	df["craigslist_email_presence"] = craigslist_email_presence(df)
	df["Phone_presence"] = Phone_presence(df)
	df["Name_presence"] = Name_presence(df)
	df["Image_presence"] = Image_presence(df)
	df["Content_word_count"] = Content_word_count(df)
	df["count_spec_chars_title"] = count_spec_chars_title(df)
	df["SpecialCharsInTitle"] = SpecialCharsInTitle(df)
	df["SpecialCharsInContent"] = SpecialCharsInContent(df)

	return df

def type_conversion():

	data_read = preprocesing_data()

	# Data type conversion of certain features to int or float
	
	df = appending_to_dataframe()
	df.is_copy = False
	df['Price'] = df['Price'].astype(int)
	df['special_chars_count_content'] = df['special_chars_count_content'].astype(int)
	df['special_chars_count_title'] = df['special_chars_count_title'].astype(int)
	df['craigslist_email_presence'] = df['craigslist_email_presence'].astype(int)
	df['personal_email'] = df['personal_email'].astype(int)
	df['external_link'] = df['external_link'].astype(int)
	df['Phone_presence'] = df['Phone_presence'].astype(int)
	df['Name_presence'] = df['Name_presence'].astype(int)
	df["Image_presence"] = df['Image_presence'].astype(int)
	df["Content_word_count"] = df["Content_word_count"].astype(int)
	df['Title Word Count'] = df['Title Word Count'].astype(int)
	df['numberOfupdates'] = df['numberOfupdates'].astype(float)
	df['PhoneAverageMarketPrice '] = df['PhoneAverageMarketPrice '].astype(float)
	df["count_spec_chars_title"] = df["count_spec_chars_title"].astype(int)
	df["SpecialCharsInTitle"] = df["SpecialCharsInTitle"].astype(int)
	df["SpecialCharsInContent"] = df["SpecialCharsInContent"].astype(int)

	return df

def training_testing_split():

	# Function to perform train-test split for modelling part

	df = type_conversion()
	y = df["Label"].tolist()

	# Splitting the undersampled dataframe in 75:25 ratio (75% training and 25% test)
	X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)
	
	return (X_train, X_test, y_train, y_test)

def feature_transformation_and_feature_union():

	# Function to combine categorical features and numerical features

	# Getting Content data for tf-idf and bigrams calculation
	get_text_data_Content = FunctionTransformer(lambda data_read: data_read['Content'], validate=False)
	# Getting Title data for tf-idf and bigrams calculation
	get_text_data_Title = FunctionTransformer(lambda data_read: data_read['Title'], validate=False)

	# Getting Entity and Numerical data together
	get_entity_data = FunctionTransformer(lambda x: x[['special_chars_count_content',
		'special_chars_count_title',
		'Content_word_count', 'Phone_presence', 'Title Word Count',
		'Image_presence', 'numberOfupdates', 'Price', 'Name_presence',
		'PhoneAverageMarketPrice ', 'craigslist_email_presence',
		"personal_email", "external_link",
		'Status_Active', 'Status_Expired', 'Status_Flagged',
       'Status_Removed', 'City_Atlanta', 'City_Baltimore', 'City_Boston',
       'City_Detroit', 'City_OrangeCounty', 'City_SanDiego ',
       'City_SanFransisco', 'City_phoenix', 'State_Arizona',
       'State_California', 'State_Georgia', 'State_Maryland',
       'State_Massachusetts', 'State_Michigan',
       'Matched Make(Averagepricefunctionality )_apple',
       'Matched Make(Averagepricefunctionality )_google',
       'Matched Make(Averagepricefunctionality )_huawei',
       'Matched Make(Averagepricefunctionality )_lg',
       'Matched Make(Averagepricefunctionality )_motorola',
       'Matched Make(Averagepricefunctionality )_samsung',
       'AdToCity(Distance Functionality)',
	'AdToPhoneArea(Distance Functionality)', 'CitytoPhoneArea(Distance Funcionality)']], validate=False)

	# Join Categorical, Entity and Numerical data together
	process_and_join_features = FeatureUnion(
            # Getting all the Entity and Numerical features together
            transformer_list = [
                ('entity_features', Pipeline([
                    ('selector1', get_entity_data)

                ])),

                #Applying TF-IDF, bigrams and stop words removal on Content data
                ('text_features_Content', Pipeline([
                    ('selector2', get_text_data_Content),
                    ('vectorizer1', TfidfVectorizer(max_df=0.5, ngram_range=(1, 2), stop_words="english"))
                ])),

                #Applying TF-IDF, bigrams and stop words removal on Title data
                ('text_features_Title', Pipeline([
                    ('selector3', get_text_data_Title),
                    ('vectorizer2', TfidfVectorizer(max_df=0.5, ngram_range=(1, 2), stop_words="english"))
                ]))
             ]
        )

	return process_and_join_features

def pipeline(process_and_join_features):

	# Function to set-up Pipeline with classifier
	clf = Pipeline([
        ('union', process_and_join_features),
        #('clf', LinearSVC(random_state=42))
        ('clf', LogisticRegression(random_state=42))
		#('clf', SGDClassifier())
        #('clf', RandomForestClassifier(random_state=42))
    	])

	return clf

def modelling(clf, X_train, y_train):
	
	# Fitting the classifier on training data
	model = clf.fit(X_train, y_train)

	# Dumping the model using joblib
	with open("craigslist_model.pickle", "wb") as craigslist_model:
		joblib.dump(model, craigslist_model)

	# Loading the model back using joblib
	with open("craigslist_model.pickle", "rb") as craigslist_model:
		pickled_model = joblib.load(craigslist_model)

	return pickled_model

def prediction(model, X_test):

	# Prediction on test data
	pred = model.predict(X_test)

	return pred


def accuracy_report_and_ROC_curve(y_test, pred):

	# Calculate precision, recall and F-1 measures of the model
	stats = classification_report(y_test, pred)
	# Calculate accuracy score of the model
	accuracy_report = accuracy_score(y_test, pred)
	# Calculating the confusion matrix of the test data
	matrix = confusion_matrix(y_test, pred)
	# Calculating True Positives
	TP = matrix[0][0]
	# Calculating False Positives
	FP = matrix[0][1]
	# Calculating False Negatives
	FN = matrix[1][0]
	# Calculating True Negatives
	TN = matrix[1][1]

	# Label Binarizing of y_test for calculating roc_auc_score
	y_test_binarize = label_binarize(y_test, classes=['not_scam', 'Scam'])
	# Label Binarizing of predicted labels for calculating roc_auc_score
	pred_binarize = label_binarize(pred, classes=['not_scam', 'Scam'])
	# Calculating roc_auc_score
	auc1 = roc_auc_score(y_test_binarize, pred_binarize)

	# Plotting the ROC Curve
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(2):
		fpr[i], tpr[i], _ = roc_curve(y_test_binarize, pred_binarize)
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarize.ravel(), pred_binarize.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	plt.figure()
	lw = 2
	plt.plot(fpr[1], tpr[1], color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")

	# Saving the ROC Curve
	plt.savefig('ROC_curve.png')

	# Returning all the required metrics and measures
	return (stats, accuracy_report, matrix, auc1, TP, FP, FN, TN)

def main():

	# Main function to call all required functions
	X_train, X_test, y_train, y_test = training_testing_split()
	process_and_join_features = feature_transformation_and_feature_union()
	clf = pipeline(process_and_join_features)
	model = modelling(clf, X_train, y_train)
	pred = prediction(model, X_test)
	
	stats, accuracy, matrix, auc, TP, FP, FN, TN = accuracy_report_and_ROC_curve(y_test, pred)

	print ("Stats on the model: \n\n", stats)
	print ("\n")
	print ("Accuracy of the model:", accuracy)
	print ("\n")
	print ("AUC score of the model:", auc)
	print ("\n")
	print ("Confusion Matrix on Test Data:", matrix)
	print ("\n")
	print ("performance measures are:")

	print ('TP' + '=' + str(TP))
	print ('FP' + '=' + str(FP))
	print ('TN' + '=' + str(TN))
	print ('FN' + '=' + str(FN))
	
	print ("=========END=========")
	print ("\n")

main()