import numpy as np
import pandas as pd
#not req  pd.set_option('display.width',1000)
pd.set_option('precision',2)

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sbn

#ignore warnings
import warnings
warnings.filterwarnings(action="ignore")

#STEP 2>Read in and Explore the Data
#*****************************************
#It's time to read in our training and testing data using pandas
#import train and test CSV files
train=pd.read_csv('train.csv')#12 columns
test=pd.read_csv('test.csv')  #11 column
print (train.describe())
print ("\n")
print (train.describe(include="all"))
print ("\n")

#STeP 3>Data Analysis
#********************************
#We're going to consider the features in the dataset and how complete they are
#get a list of the features within the dataset
print ("\n\n",train.columns)
print()
print (train.head())
print()
print (train.sample())
print ("Data types for each feature:-")
print (train.dtypes)
print()
print (pd.isnull(train).sum())  #bradcats on all training data then every column sum where it get null values

#We can see that except for the above mentioned missing values,m

#4>Data Visualization
#********************************
#It's time to visualize our data so we can estimate few predictions
#4(A):- Sex feature:-
#-----------------------------
sbn.barplot(x="Sex",y="Survived",data=train)
plt.show()

#print percentages of females vs. males that survive
#print "Percentage of females who survive:",train["Survived"][train["Sex"]=='female'].value_counts(normalized==True)[1]*100

print ("-----------------\n\n")
print (train)

print ("-----------------\n\n")
print (train["Survived"])

print ("-----------------\n\n")
print (train["Sex"]=='female')    #Array operation broadcast at 891 where true represents female

print ("*******************\n\n")
print (train["Survived"][train["Sex"]=='female']) #survival status of females only

print ("*******************\n\n")
print (train["Survived"][train["Sex"]=='female'].value_counts())  #count how many 0s and how many 1s

print ("===================\n\n")
print (train["Survived"][train["Sex"]=='female'].value_counts(normalize=True))  #fraction of 1 ratio survival/total

print (train["Survived"][train["Sex"]=='female'].value_counts(normalize=True)[1])

print()
print ("Percentage of females who survived:",train["Survived"][train["Sex"]=='female'].value_counts(normalize=True)[1]*100)
print ("Percentage of males who survived:",train["Survived"][train["Sex"]=='male'].value_counts(normalize=True)[1]*100)
print()
#---------------------
#4.B)Pclass Feature
#---------------------
#draw a bar plot of survival by Pclass
sbn.barplot(x="Pclass",y="Survived",data=train)

#print percentage of people by Pclass that survived
print ("Percentage of Pclass=1 who survived:",train["Survived"][train["Pclass"]==1].value_counts(normalize=True)[1]*100)
print ("Percentage of Pclass=2 who survived:",train["Survived"][train["Pclass"]==2].value_counts(normalize=True)[1]*100)
print ("Percentage of Pclass=3 who survived:",train["Survived"][train["Pclass"]==3].value_counts(normalize=True)[1]*100)
print(0)
plt.show()
#---------------------
#4.C)SibSp Feature
#---------------------
#draw a bar plot of survival by SibSp vs. survival
sbn.barplot(x="SibSp",y="Survived",data=train)

#I won't be printing indivisual percent values for all of these
print ("Percentage of SibSp=0 who survived:",train["Survived"][train["SibSp"]==0].value_counts(normalize=True)[1]*100)
print ("Percentage of SibSp=1 who survived:",train["Survived"][train["SibSp"]==1].value_counts(normalize=True)[1]*100)
print ("Percentage of SibSp=2 who survived:",train["Survived"][train["SibSp"]==2].value_counts(normalize=True)[1]*100)
print()
plt.show()
#---------------------
#4.D)Parch Feature
#---------------------
#draw a bar plot of survival by Parch vs. survival
sbn.barplot(x="Parch",y="Survived",data=train)

plt.show()
#Some Observations from Above Output
#-------------------------------------
#People with less than four parents or children aboard are more likely to survive
#Again,people travelling alone are less likely to survive than those with a sibling or spouse

#---------------------
#4.E)Age Feature
#---------------------
#sort the ages into logical categories
train["Age"]=train["Age"].fillna(-0.5)   #filled the missing values of age column by -0.5
test["Age"]=test["Age"].fillna(-0.5)

bins=[-1,0,5,12,18,24,35,60,np.inf]      #np.inf:- positive infiniy
labels=['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
train['AgeGroup']=pd.cut(train["Age"],bins,labels=labels) #categorize and check data category in which bins
test['AgeGroup']=pd.cut(test["Age"],bins,labels=labels)
print (train)
#draw a bar plot of Age vs.Survival
sbn.barplot(x="AgeGroup",y="Survived",data=train)
plt.show()

#Done*********************************************************



#Some Observations from above output
#------------------------------------
#Babies are more likely to survive than any other age group


#----------------------
#4.F)Cabin Feature
#----------------------

#I think the idea here is that people with recorded cabin numbers are of socio economic importance
#and thus more likely to survive.

train["CabinBool"]=(train["Cabin"].notnull().astype('int'))  #204 passengers have notnull value for cabin.so 204 passengers will have true value
test["CabinBool"]=(test["Cabin"].notnull().astype('int'))

print ("########################################\n\n")
print (train)

#calculate percentages of CabinBool vs. survived
print("Percentage ofCabinBool=1 who survived:",train["Survived"][train["CabinBool"]==1].value_counts(normalize=True)[1]*100)
print("Percentage ofCabinBool=0 who survived:",train["Survived"][train["CabinBool"]==0].value_counts(normalize=True)[1]*100)
#draw a bar plot of CabinBool vs.survival
sbn.barplot(x="CabinBool",y="Survived",data=train)
plt.show()


#5)Cleaning Data
#**************************

#Time to clean our data to account for missing values and unnecessary iinformation

#Looking at the Test Data
#Let's see how our test data looks!

print (test.describe(include="all")) #prints about non-numeric data also
print (pd.isnull(test).sum())

#Cabin Feature
#We'll start off by dropping the Cabin Feature since it is no more useful
train=train.drop(['Cabin'],axis=1)
test=test.drop(['Cabin'],axis=1)

#Ticket Feature
#We can also drop the Ticket Feature since it is unlikely to yield any useful information
train=train.drop(['Ticket'],axis=1)
test=test.drop(['Ticket'],axis=1)

#Embarked Feature
#now we need to fill in the missing values in the Embarked Feature
print ("Number of people embarking in Southampton(S):",)#comma means we have to print more

print ("\nSHAPE",train[train["Embarked"]=='S'].shape)
print ("SHAPE[0]",train[train["Embarked"]=='S'].shape[0])#this gives number of rows,that is number of people boarding from S

print ("Number of people embarking in Queenstown(Q):",)

print ("\nSHAPE",train[train["Embarked"]=="Q"].shape)
queenstown=train[train["Embarked"]=="Q"].shape[0]
print (queenstown)

print ("Number pf people embarking in Cherbourg(C):")
cherbourg=train[train["Embarked"]=='C'].shape[0]
print (cherbourg)

#It's clear that the majority of people embarked in Southampton
#Let's go ahead and fill in the missing values with S

#replacing missing values in the Embarked feature with S
train["Embarked"]=train["Embarked"].fillna("S")   #filled the missing values of age column by S
test["Embarked"]=test["Embarked"].fillna("S")     #another method:train=train.fillna({"Embarked":"S"})

#Age Feature
#Next we'll fill in the missing values in the Age feature
#Since a  higher percentage of values are missing,
#it would be illogical to fill all of them with the same values
#Instead,let's try to find a way to predict missing ages

#create a combined group of botyh datasets
combine=[train,test]
print (combine[0])  #to print first value train of array list combine

#extract a title for each Name in the train and test datasets
for dataset in combine:  #combine has two values so this loop will work twice
    dataset['Title']=dataset['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)  #+extracts one or more from previous expression

print ("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print (train)
print()

print (pd.crosstab(train['Title'],train['Sex']))

#OUTPUT

#Sex       female  male
#Title
#Capt           0     1
#Col            0     2
#Countess       1     0
#Don            0     1
#Dr             1     6
#Jonkheer       0     1
#Lady           1     0
#Major          0     2
#Master         0    40
#Miss         182     0
#Mlle           2     0
#Mme            1     0
#Mr             0   517
#Mrs          125     0
#Ms             1     0
#Rev            0     6
#Sir            0     1

#replace various titles with more common names
for dataset in combine:
    dataset['Title']=dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
    #replace all these by rare if found
    dataset['Title'] = dataset['Title'].replace(['Countess','Sir'],'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')  #replace ms by miss
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')  #replace mme by mrs

print ("\n\nAfter grouping rare title:\n",train)
print (pd.crosstab(train['Title'],train['Sex']))

'''
After grouping rare title:
     PassengerId  Survived  Pclass   ...       AgeGroup CabinBool   Title
0              1         0       3   ...        Student         0      Mr
1              2         1       1   ...          Adult         1     Mrs
2              3         1       3   ...    Young Adult         0    Miss
3              4         1       1   ...    Young Adult         1     Mrs
4              5         0       3   ...    Young Adult         0      Mr
5              6         0       3   ...        Unknown         0      Mr
6              7         0       1   ...          Adult         1      Mr
7              8         0       3   ...           Baby         0  Master
8              9         1       3   ...    Young Adult         0     Mrs
9             10         1       2   ...       Teenager         0     Mrs
10            11         1       3   ...           Baby         1    Miss
11            12         1       1   ...          Adult         1    Miss
12            13         0       3   ...        Student         0      Mr
13            14         0       3   ...          Adult         0      Mr
14            15         0       3   ...       Teenager         0    Miss
15            16         1       2   ...          Adult         0     Mrs
16            17         0       3   ...           Baby         0  Master
17            18         1       2   ...        Unknown         0      Mr
18            19         0       3   ...    Young Adult         0     Mrs
19            20         1       3   ...        Unknown         0     Mrs
20            21         0       2   ...    Young Adult         0      Mr
21            22         1       2   ...    Young Adult         1      Mr
22            23         1       3   ...       Teenager         0    Miss
23            24         1       1   ...    Young Adult         1      Mr
24            25         0       3   ...          Child         0    Miss
25            26         1       3   ...          Adult         0     Mrs
26            27         0       3   ...        Unknown         0      Mr
27            28         0       1   ...        Student         1      Mr
28            29         1       3   ...        Unknown         0    Miss
29            30         0       3   ...        Unknown         0      Mr
..           ...       ...     ...   ...            ...       ...     ...
861          862         0       2   ...        Student         0      Mr
862          863         1       1   ...          Adult         1     Mrs
863          864         0       3   ...        Unknown         0    Miss
864          865         0       2   ...        Student         0      Mr
865          866         1       2   ...          Adult         0     Mrs
866          867         1       2   ...    Young Adult         0    Miss
867          868         0       1   ...    Young Adult         1      Mr
868          869         0       3   ...        Unknown         0      Mr
869          870         1       3   ...           Baby         0  Master
870          871         0       3   ...    Young Adult         0      Mr
871          872         1       1   ...          Adult         1     Mrs
872          873         0       1   ...    Young Adult         1      Mr
873          874         0       3   ...          Adult         0      Mr
874          875         1       2   ...    Young Adult         0     Mrs
875          876         1       3   ...       Teenager         0    Miss
876          877         0       3   ...        Student         0      Mr
877          878         0       3   ...        Student         0      Mr
878          879         0       3   ...        Unknown         0      Mr
879          880         1       1   ...          Adult         1     Mrs
880          881         1       2   ...    Young Adult         0     Mrs
881          882         0       3   ...    Young Adult         0      Mr
882          883         0       3   ...        Student         0    Miss
883          884         0       2   ...    Young Adult         0      Mr
884          885         0       3   ...    Young Adult         0      Mr
885          886         0       3   ...          Adult         0     Mrs
886          887         0       2   ...    Young Adult         0    Rare
887          888         1       1   ...        Student         1    Miss
888          889         0       3   ...        Unknown         0    Miss
889          890         1       1   ...    Young Adult         1      Mr
890          891         0       3   ...    Young Adult         0      Mr
'''
#Sex     female  male
#Title
#Dr           1     6
#Master       0    40
#Miss       185     0
#Mr           0   517
#Mrs        126     0
#Rare         1    13
#Royal        1     1

print (train[['Title','Survived']].groupby(['Title'],as_index=False).count())   #will print only title and survive column grouping will be done on the basis of title
#asindex=False so that index does not print

#    Title  Survived
#0      Dr         7
#1  Master        40
#2    Miss       185
#3      Mr       517
#4     Mrs       126
#5    Rare        14
#6   Royal         2

print ("\nMap each of the title groups to a numerical value")
title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Royal":5,"Rare":6}  #text column cannot be used so converting useful column to numeric
for dataset in combine:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)

print ("\nAfter replacing title with numeric value")
print (train)
print()

#OUTPUT
'''
Map each of the title groups to a numerical value
After replacing title with numeric value
     PassengerId  Survived  Pclass  ...       AgeGroup CabinBool  Title
0              1         0       3  ...        Student         0    1.0
1              2         1       1  ...          Adult         1    3.0
2              3         1       3  ...    Young Adult         0    2.0
3              4         1       1  ...    Young Adult         1    3.0
4              5         0       3  ...    Young Adult         0    1.0
5              6         0       3  ...        Unknown         0    1.0
6              7         0       1  ...          Adult         1    1.0
7              8         0       3  ...           Baby         0    4.0
8              9         1       3  ...    Young Adult         0    3.0
9             10         1       2  ...       Teenager         0    3.0
10            11         1       3  ...           Baby         1    2.0
11            12         1       1  ...          Adult         1    2.0
12            13         0       3  ...        Student         0    1.0
13            14         0       3  ...          Adult         0    1.0
14            15         0       3  ...       Teenager         0    2.0
15            16         1       2  ...          Adult         0    3.0
16            17         0       3  ...           Baby         0    4.0
17            18         1       2  ...        Unknown         0    1.0
18            19         0       3  ...    Young Adult         0    3.0
19            20         1       3  ...        Unknown         0    3.0
20            21         0       2  ...    Young Adult         0    1.0
21            22         1       2  ...    Young Adult         1    1.0
22            23         1       3  ...       Teenager         0    2.0
23            24         1       1  ...    Young Adult         1    1.0
24            25         0       3  ...          Child         0    2.0
25            26         1       3  ...          Adult         0    3.0
26            27         0       3  ...        Unknown         0    1.0
27            28         0       1  ...        Student         1    1.0
28            29         1       3  ...        Unknown         0    2.0
29            30         0       3  ...        Unknown         0    1.0
..           ...       ...     ...  ...            ...       ...    ...
861          862         0       2  ...        Student         0    1.0
862          863         1       1  ...          Adult         1    3.0
863          864         0       3  ...        Unknown         0    2.0
864          865         0       2  ...        Student         0    1.0
865          866         1       2  ...          Adult         0    3.0
866          867         1       2  ...    Young Adult         0    2.0
867          868         0       1  ...    Young Adult         1    1.0
868          869         0       3  ...        Unknown         0    1.0
869          870         1       3  ...           Baby         0    4.0
870          871         0       3  ...    Young Adult         0    1.0
871          872         1       1  ...          Adult         1    3.0
872          873         0       1  ...    Young Adult         1    1.0
873          874         0       3  ...          Adult         0    1.0
874          875         1       2  ...    Young Adult         0    3.0
875          876         1       3  ...       Teenager         0    2.0
876          877         0       3  ...        Student         0    1.0
877          878         0       3  ...        Student         0    1.0
878          879         0       3  ...        Unknown         0    1.0
879          880         1       1  ...          Adult         1    3.0
880          881         1       2  ...    Young Adult         0    3.0
881          882         0       3  ...    Young Adult         0    1.0
882          883         0       3  ...        Student         0    2.0
883          884         0       2  ...    Young Adult         0    1.0
884          885         0       3  ...    Young Adult         0    1.0
885          886         0       3  ...          Adult         0    3.0
886          887         0       2  ...    Young Adult         0    6.0
887          888         1       1  ...        Student         1    2.0
888          889         0       3  ...        Unknown         0    2.0
889          890         1       1  ...    Young Adult         1    1.0
890          891         0       3  ...    Young Adult         0    1.0
'''

#we will calculate mode of age for each title so that we can predict the age group of unknown age
mr_age=train[train["Title"]==1]["AgeGroup"].mode()  #1 gets applied on all values of title column
print ("mode() of mr_age:",mr_age)                          #Mr.Young Adult

miss_age=train[train["Title"]==2]["AgeGroup"].mode()      #Miss=student
print ("mode() of miss_age:",miss_age)

mrs_age=train[train["Title"]==3]["AgeGroup"].mode()       #Mrs.=Adult
print ("mode() of mrs_age:",mrs_age)

master_age=train[train["Title"]==4]["AgeGroup"].mode()    #Baby
print ("mode() of master_age:",master_age)

royal_age=train[train["Title"]==5]["AgeGroup"].mode()     #Young Adult and Adult have same frequency
print ("mode() of royal_age:",royal_age)

rare_age=train[train["Title"]==6]["AgeGroup"].mode()      #Adult
print ("mode() of rare_age:",rare_age)
print()

print ("\n ************************************* \n")
print ("\n *******   train[Age Group][0]: \n")
for x in range(10):                      #from 1 to 10 we want to print
    print (train["AgeGroup"][x])

age_title_mapping={1:"Young Adult",2:"Student",3:"Adult",4:"Baby",5:"Adult",6:"Adult"}
for x in range(len(train["AgeGroup"])):                         #loop works for 0 to 890
    if train["AgeGroup"][x]=="Unknown":                         #x=5(means for 6th record)
        train["AgeGroup"][x]=age_title_mapping[train["Title"][x]]
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x]=="Unknown":
        test["AgeGroup"][x]=age_title_mapping[test["Title"][x]]
print (train["AgeGroup"])
age_mapping={"Baby":1,"Child":2,"Teenager":3,"Student":4,"Young Adult":5,"Adult":6,"Senior":7}
train["AgeGroup"]=train['AgeGroup'].map(age_mapping)
test["AgeGroup"]=test['AgeGroup'].map(age_mapping)

#dropping the age feature now,might change
train=train.drop(['Age'],axis=1)
test=test.drop(['Age'],axis=1)

print ("\n\nAge column dropped")
print (train)

#Name feature
#We can drop the name feature now that we've extracted the titles

#drop the name feature since it contains no more useful information
train=train.drop(['Name'],axis=1)
test=test.drop(['Name'],axis=1)

#Sex feature
#map each sex value to a numerical value
sex_mapping={"male":0,"female":1}
train['Sex']=train['Sex'].map(sex_mapping)
test['Sex']=test['Sex'].map(sex_mapping)

print (train)

#Embarked Feature
#map each Embarked value to a numerical value
embarked_mapping={"S":1,"C":2,"Q":3}
train['Embarked']=train['Embarked'].map(embarked_mapping)
test['Embarked']=test['Embarked'].map(embarked_mapping)
print()
print (train.head())

#Fare Feature
#It is time separate the fare values into some logical groups as well
#filling in the single value in the test dataset

#filling in missing Fare value in test set based on mean fare for that Pclass
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass=test["Pclass"][x]  #Pclass=3
        test["Fare"][x]=round(train[train["Pclass"]==pclass]["Fare"].mean(),2) #rounding off mean to 2 decimal places

#map Fare values into groups of numerical values
train['FareBand']=pd.qcut(train['Fare'],4,labels=[1,2,3,4])
test['FareBand']=pd.qcut(test['Fare'],4,labels=[1,2,3,4])

#drop Fare values
train=train.drop(['Fare'],axis=1)
test=test.drop(['Fare'],axis=1)
#check train data
print ("\n\nFare column dropped\n")
print (train)

#check test data
print()
print (test.head())

#***************************************
#6)Choosing the Best Model
#***************************************

#Splitting the training data
#We will use part of our training data (20% in this case) to test the accuracy of algorithm

from sklearn.model_selection import train_test_split

input_predictors=train.drop(['Survived','PassengerId'],axis=1)
output_target=train["Survived"]
x_train,x_val,y_train,y_val=train_test_split(input_predictors,output_target,test_size=0.20,random_state=7)

#For each model,we set the model,fit it with 80%of our training data
#predict for 20 % of the training data and check the accuracy

from sklearn.metrics import accuracy_score
#MODEL-1)LogisticRegression
#---------------------------------
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_val)
acc_logreg=round(accuracy_score(y_pred,y_val)*100,2)
print ("Model-1: Accuracy of LogisticRegression:",acc_logreg)

#MODEL-2)Gaussian Naive Bayes
#---------------------------------
from sklearn.naive_bayes import GaussianNB
gaussian=GaussianNB()
gaussian.fit(x_train,y_train)
y_pred=gaussian.predict(x_val)
acc_gaussian=round(accuracy_score(y_pred,y_val)*100,2)
print ("Model-2: Accuracy of GaussianNB:",acc_gaussian)

#MODEL-3)Support Vector Machines
#---------------------------------
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_val)
acc_svc=round(accuracy_score(y_pred,y_val)*100,2)
print ("Model-3: Accuracy of Support Vector Machines:",acc_svc)

#MODEL-4) Linear SVC
from sklearn.svm import LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-4: Accuracy of LinearSVC : ",acc_linear_svc)

#MODEL-5) Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-5: Accuracy of Perceptron : ",acc_perceptron)

#MODEL-6) Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-6: Accuracy of DecisionTreeClassifier : ", acc_decisiontree)

#MODEL-7) Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-7: Accuracy of RandomForestClassifier : ",acc_randomforest)

#MODEL-8) KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-8: Accuracy of k-Nearest Neighbors : ",acc_knn)

#MODEL-9) Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-9: Accuracy of Stochastic Gradient Descent : ",acc_sgd)

#MODEL-10) Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-10: Accuracy of GradientBoostingClassifier : ",acc_gbk)

#Let us compare the accuracies of each model
models=pd.DataFrame({'Model':['Logistic Regression','Gaussian Naive Bayes','Support Vector Machines','Linear SVC','Perceptron',
                              'Decision Tree Classifier','Random Forest','KNN','Stochastic Gradient Descent','Gradient Boosting Classifier'],
                     'Score':[acc_logreg,acc_gaussian,acc_svc,acc_linear_svc,acc_perceptron,acc_decisiontree,
                              acc_randomforest,acc_knn,acc_sgd,acc_gbk]})
print()
print (models.sort_values(by='Score',ascending=False))

#7)Creating Submission Result File
#************************************
#It is time to create a submission.csv file which includes our prediction

#set ids as PassengerId and predict survival
ids=test['PassengerId']
predictions=randomforest.predict(test.drop('PassengerId',axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output=pd.DataFrame({'PassengerId':ids,' Survived':predictions})

print ("All survival predictions done")
print ("All predictions exported to submission.csv file")

print (output)

