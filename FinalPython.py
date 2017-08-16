
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import model_selection  
from sklearn import linear_model

def fit_log_model(Xtrain,Ytrain,Xtest,Ytest):
    model = linear_model.LogisticRegression()
    if(len(Xtrain.columns)>1): 
        model.fit(Xtrain,Ytrain)
    elif (len(Xtrain.columns)==1):
        model.fit(Xtrain.values.reshape(-1,1),Ytrain)
        return model
    else: print("Input needs to be nonempty DataFrame")
    score = model.score(Xtest,Ytest)
    print("Accuracy of Logistic Regression using "+str(len(Xtrain.columns))+" features: "+str(score))
    return model

def fit_lin_model(Xtrain,Ytrain,Xtest,Ytest):
    model = linear_model.LinearRegression()
    if(len(Xtrain.columns)>1): 
        model.fit(Xtrain,Ytrain)
    elif (len(Xtrain.columns)==1):
        model.fit(Xtrain.values.reshape(-1,1),Ytrain)
    else: print("Input needs to be nonempty DataFrame")
    score = model.score(Xtest,Ytest)
    print("R-Squared: "+str(score))
    return model   


def cross_validate(xtrain,ytrain):
    results = []
    names = []
    for  column in xtrain:
        # Performing 10-fold cross-validation
        kfold = model_selection.KFold(n_splits=10)
        # Note: using training dataset
        cv_results = model_selection.cross_val_score(model, xtrain[column].values.reshape(-1,1), ytrain, cv=kfold)
        results.append(cv_results)
        names.append(column)
        msg = "%s: %f (%f)" % (column, cv_results.mean(), cv_results.std())
        print(msg)
        
def create_linear_graphs(Xtrain,Ytrain,Xtest,Ytest):
    for column in Xtrain:
        model = linear_model.LinearRegression()
        model.fit(Xtrain[column].values.reshape(-1,1),Ytrain)
        preds = model.predict(Xtest[column].values.reshape(-1,1))
        plt.plot(Xtest[column].values.reshape(-1,1),preds,'bo',alpha = .2)
        plt.plot(Xtest[column].values.reshape(-1,1),Ytest,'ro', alpha = .2)
        plt.xlabel(column[0].upper()+column[1:])
        plt.ylabel("Quality")
        plt.show()
        score = model.score(Xtest[column].values.reshape(-1,1),Ytest)
        print("R-Squared: "+str(score))
        
def funcGetDataFromURLAndFormat(url):
    if url == '':
        return 'empty URL'
    df = pd.read_csv(url, header = 0, delimiter = ';')
    df.columns = [name.replace(' ','_') for name in df.columns]
    return df

# Load dataset
# White wines first
white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
white_df = funcGetDataFromURLAndFormat(white_url)
# Then read wines
red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
red_df = funcGetDataFromURLAndFormat(red_url)

all_wines = white_df.append(red_df)

white_df["Type"]= 0
red_df["Type"]= 1 #encoding




# LOOK FOR OUTLIERS
# WHITE WINES FIRST
# Take the mean and standard deviation of each column -- White wines
mean_and_std = [[np.mean(white_df[col]), np.std(white_df[col])] for col in white_df.columns]
# 3 standard deviations from the mean
lower_bound = [row[0] - 3 * row[1] for row in mean_and_std]
upper_bound = [row[0] + 3 * row[1] for row in mean_and_std]
# Points outside this range for each column
fixed_acidity_outliers = white_df[(white_df.fixed_acidity < lower_bound[0]) | (white_df.fixed_acidity > upper_bound[0])]
volatile_acidity_outliers = white_df[(white_df.volatile_acidity < lower_bound[1]) | (white_df.volatile_acidity > upper_bound[1])]
citric_acid_outliers = white_df[(white_df.citric_acid < lower_bound[2]) | (white_df.citric_acid > upper_bound[2])]
residual_sugar_outliers = white_df[(white_df.residual_sugar < lower_bound[3]) | (white_df.residual_sugar > upper_bound[3])]
chlorides_outliers = white_df[(white_df.chlorides < lower_bound[4]) | (white_df.chlorides > upper_bound[4])]
free_sulfur_dioxide_outliers = white_df[(white_df.free_sulfur_dioxide < lower_bound[5]) | (white_df.free_sulfur_dioxide > upper_bound[5])]
total_sulfur_dioxide_outliers = white_df[(white_df.total_sulfur_dioxide < lower_bound[6]) | (white_df.total_sulfur_dioxide > upper_bound[6])]
density_outliers = white_df[(white_df.density < lower_bound[7]) | (white_df.density > upper_bound[7])]
pH_outliers = white_df[(white_df.pH < lower_bound[8]) | (white_df.pH > upper_bound[8])]
sulphates_outliers = white_df[(white_df.sulphates < lower_bound[9]) | (white_df.sulphates > upper_bound[9])]
alcohol_outliers = white_df[(white_df.alcohol < lower_bound[10]) | (white_df.alcohol > upper_bound[10])]
quality_outliers = white_df[(white_df.quality < lower_bound[11]) | (white_df.quality > upper_bound[11])]
# Do a merge to look for wines outside the 3 stdev range that are present in another column's wines
# This gives us things that we've decided are outliers
outliers_list = [fixed_acidity_outliers, volatile_acidity_outliers, citric_acid_outliers, residual_sugar_outliers, chlorides_outliers, 
 free_sulfur_dioxide_outliers, total_sulfur_dioxide_outliers, density_outliers, pH_outliers, sulphates_outliers,
 alcohol_outliers, quality_outliers] # Make a list
white_duplicate_outliers = pd.DataFrame()
for first_element in range(len(outliers_list)-1): # Iterate through the list
    for second_element in range(first_element+1,len(outliers_list) ):
        white_duplicate_outliers = white_duplicate_outliers.append(outliers_list[first_element].merge(outliers_list[second_element]))
# RED WINES NEXT
mean_and_std = [[np.mean(red_df[col]), np.std(red_df[col])] for col in red_df.columns]
# 3 standard deviations from the mean
lower_bound = [row[0] - 3 * row[1] for row in mean_and_std]
upper_bound = [row[0] + 3 * row[1] for row in mean_and_std]
# Points outside this range for each column
fixed_acidity_outliers = red_df[(red_df.fixed_acidity < lower_bound[0]) | (red_df.fixed_acidity > upper_bound[0])]
volatile_acidity_outliers = red_df[(red_df.volatile_acidity < lower_bound[1]) | (red_df.volatile_acidity > upper_bound[1])]
citric_acid_outliers = red_df[(red_df.citric_acid < lower_bound[2]) | (red_df.citric_acid > upper_bound[2])]
residual_sugar_outliers = red_df[(red_df.residual_sugar < lower_bound[3]) | (red_df.residual_sugar > upper_bound[3])]
chlorides_outliers = red_df[(red_df.chlorides < lower_bound[4]) | (red_df.chlorides > upper_bound[4])]
free_sulfur_dioxide_outliers = red_df[(red_df.free_sulfur_dioxide < lower_bound[5]) | (red_df.free_sulfur_dioxide > upper_bound[5])]
total_sulfur_dioxide_outliers = red_df[(red_df.total_sulfur_dioxide < lower_bound[6]) | (red_df.total_sulfur_dioxide > upper_bound[6])]
density_outliers = red_df[(red_df.density < lower_bound[7]) | (red_df.density > upper_bound[7])]
pH_outliers = red_df[(red_df.pH < lower_bound[8]) | (red_df.pH > upper_bound[8])]
sulphates_outliers = red_df[(red_df.sulphates < lower_bound[9]) | (red_df.sulphates > upper_bound[9])]
alcohol_outliers = red_df[(red_df.alcohol < lower_bound[10]) | (red_df.alcohol > upper_bound[10])]
quality_outliers = red_df[(red_df.quality < lower_bound[11]) | (red_df.quality > upper_bound[11])]
# Do a merge to look for wines outside the 3 stdev range that are present in another column's wines
# This gives us things that we've decided are outliers
outliers_list = [fixed_acidity_outliers, volatile_acidity_outliers, citric_acid_outliers, residual_sugar_outliers, chlorides_outliers, 
 free_sulfur_dioxide_outliers, total_sulfur_dioxide_outliers, density_outliers, pH_outliers, sulphates_outliers,
 alcohol_outliers, quality_outliers] # Make a list
red_duplicate_outliers = pd.DataFrame()
for first_element in range(len(outliers_list)-1): # Iterate through the list
    for second_element in range(first_element+1,len(outliers_list) ):
        red_duplicate_outliers = red_duplicate_outliers.append(outliers_list[first_element].merge(outliers_list[second_element]))
# QUERIES OF THE DATA THAT WE ARE INTERESTED IN
# 1 -- What are the highest-rated wines? We want to look at these wines to see what theyre doing well
best_wines = all_wines[(all_wines.quality>8)]
# Write to a table
best_wines.to_csv('best_wines.csv')
# 2 -- What are the worst-rated wines? We want to look at these wines to see what theyre doing poorly
worst_wines = all_wines[(all_wines.quality<4)]
# Write to a table
worst_wines.to_csv('worst_wines.csv')
# 3 -- Now figure out how many of each wines fall into a specific quality
quality_counts = all_wines.groupby(['quality']).count()['alcohol']
# Plot 
quality_counts.plot(kind = 'bar')
plt.show()
# 4 -- Now interested in how many of each rating by wine color
quality_counts_red_norm = red_df.groupby(['quality']).count()['alcohol']/len(red_df)
quality_counts_white_norm = white_df.groupby(['quality']).count()['alcohol']/len(white_df)
# Plot 
quality_counts_red_norm.plot(kind = 'bar')
plt.show()
quality_counts_white_norm.plot(kind = 'bar')
plt.show()


WhiteXTrain1,WhiteXTest1, WhiteYTrain1, WhiteYTest1 = model_selection.train_test_split(white_df.drop("Type",axis =1),white_df["Type"])
RedXTrain1,RedXTest1, RedYTrain1, RedYTest1 = model_selection.train_test_split(red_df.drop("Type",axis =1),red_df["Type"])
WhiteXTrain2,WhiteXTest2, WhiteYTrain2, WhiteYTest2 = model_selection.train_test_split(white_df.drop(["Type","quality"],axis =1),white_df["quality"])
RedXTrain2,RedXTest2, RedYTrain2, RedYTest2 = model_selection.train_test_split(red_df.drop(["Type","quality"],axis =1),red_df["quality"])

#print(allwine.drop("Type",axis = 1))
Xtrain1 = WhiteXTrain1.append(RedXTrain1)
Ytrain1 = WhiteYTrain1.append(RedYTrain1)
Xtest1 = WhiteXTest1.append(RedXTest1)
Ytest1 = WhiteYTest1.append(RedYTest1)

WhiteXTrain2 = WhiteXTrain2.sample(1599,random_state = 1)
WhiteYTrain2 = WhiteYTrain2.sample(1599,random_state = 1)




# pd.plotting.scatter_matrix(allwine, figsize=(22,22))
# plt.show()

print("-------")
print("White Wine Regression with All Variables:")
model = fit_lin_model(WhiteXTrain2,WhiteYTrain2,WhiteXTest2,WhiteYTest2)
print("-------")
print("Red Wine Regression with All Variables:")
model = fit_lin_model(RedXTrain2,RedYTrain2,RedXTest2,RedYTest2)
print("-------")
print("One Feature Linear Regression")
print("-------")
print("White Wines: ")
cross_validate(WhiteXTrain2,WhiteYTrain2)
print("-------")
print("Red Wines: ")
cross_validate(RedXTrain2,RedYTrain2)
print("-------")


for column in Xtrain1:
    plt.hist(white_df[column])
    plt.xlabel("White: "+column[0].upper()+column[1:])
    plt.ylabel("Frequency")
    plt.show()
    plt.hist(red_df[column])
    plt.xlabel("Red : "+column[0].upper()+column[1:])
    plt.ylabel("Frequency")
    plt.show()
    

create_linear_graphs(RedXTrain2,RedYTrain2,RedXTest2,RedYTest2)
create_linear_graphs(WhiteXTrain2,WhiteYTrain2,WhiteXTest2,WhiteYTest2)





print("-------")
model = fit_log_model(Xtrain1,Ytrain1,Xtest1,Ytest1)
print("-------")
print("One Feature Logistic Models:")
cross_validate(Xtrain1, Ytrain1)
print("-------")



for column in Xtrain1:
    model = fit_log_model(Xtrain1[column].to_frame(),Ytrain1,Xtest1,Ytest1)
    preds = model.predict_proba(Xtest1[column].values.reshape(-1,1))
    preds = [pred[1] for pred in preds]
    plt.plot(Xtest1[column].values.reshape(-1,1),preds,'bo',alpha = .2)
    plt.plot(Xtest1[column].values.reshape(-1,1),Ytest1,'ro', alpha = .2)
    plt.axhline(y=.5)
    plt.xlabel(column[0].upper()+column[1:])
    plt.ylabel("Probabilty Wine is Red")
    score = model.score(Xtest1[column].values.reshape(-1,1),Ytest1)
    plt.show()
    print("Accuracy: "+str(score))


print("-------")
model = fit_log_model(Xtrain1[["total_sulfur_dioxide","volatile_acidity"]],Ytrain1,Xtest1[["total_sulfur_dioxide","volatile_acidity"]],Ytest1)


