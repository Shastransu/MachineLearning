#Final Project
''' machine learning project description:
Titanic survival prediction:-
Inn this challenge,we ask you to complete the analysis of what sorts of people were likely to survive.
Overview
1)training set(train.csv) 891 Rows
2)test set(test.csv)      418 rows
No of colums
1:- Passenger ID
2:- Survived
3:-Name
4:-Sex
5:-Age
6:-Sibsp
7:-parch
8:-ticket
9:-fare
10:-cabin
11:-embarked
12:-pclass

Contents:-
1. Import necessary Libraries
2. Read In and Explore the historic data
3.Data Analysis
4.Data Visulaization
5.Cleaning data
6.Choosing the best model
7.Creating Submission file
'''
#data analysis libraries
import numpy as np        #used for stroring large multidimensional array
import pandas as pd       #flexible data manipulation,manupulating data tables
#pd.set_option('display.width',1000)   #Not required now
pd.set_option('precision',2)  #print upto two decimal places of float data

#visulaization libraries
import matplotlib.pyplot as plt               #General graphs(bars,pies,scatter plots)
import seaborn as sbn                         #Statstical data visualization based on matplotlib

#ignore warning
import warnings
warnings.filterwarnings('ignore')

#Step-2)Read in and explore the data
#***************************************
#It's time to read in our training and testing data using

#import train and test CSV files
train=pd.read_csv('train.csv')#12 columns
test=pd.read_csv('test.csv')  #11 column
print (train.describe())
print ("\n")
print (train.describe(include="all"))
print ("\n")

#Step 3) Data Analysis
#*******************************************
# we are going to consider the features in the dataset and how comlete they are

print ("\n\n",train.columns)
print()
print (train.head())
print()
print (train.sample())
print ("Data types for each feature:-")
print (train.dtypes)

#Some observation from the output

print()
print(pd.isnull(train).sum() )   #bradcast on all training data then every column sum where it get null value

#We can see that except for the above mentioned missing values

#Relationship between features and survival

#4 data visulaization
#***************************************************
#its time to visulization our data so we can estimate predictoions
#4(A):- Sex feature

sbn.barplot(x="Sex",y="Survived",data=train)   #people of those gender which survived (relation of gender and survival)
plt.show()

#print number of females vs.males that survive
print("_________________\n\n")
print(train)
#print number of females vs.males that survive:"train["Survived"][train]["Sex"]=='female'].value_counts(normalize=True)[1]*100
print("_________________\n\n")
print (train["Survived"])

print("---------------\n\n")
print(train["Sex"]=='female')      #Array operation broadcast at 891  where true represents female

print("******************\n\n")
print (train["Survived"][train["Sex"]=='female']) #all female survival status 1 means survied and 0 means not survived


print("*******************\n\n")
print (train["Survived"][train["Sex"]=='female'].value_counts())  #Count how many 0 and how many 1


print("*******************\n\n")
print (train["Survived"][train["Sex"]=='female'].value_counts(normalize=True))  #fraction of 1 ratio survival/total

print (train["Survived"][train["Sex"]=='female'].value_counts(normalize=True)[1])
print()

print ("Percentage of females who survived:",train["Survived"][train["Sex"]=='female'].value_counts(normalize=True)[1]*100)
print ("Percentage of males who survived:",train["Survived"][train["Sex"]=='male'].value_counts(normalize=True)[1]*100)
#----------------------
#4.B) Pclass Feature
#-------------------------
#draw a bar plot of survival by Pclass
sbn.barplot(x="Pclass",y="Survived",data=train)   #people of those gender which survived (relation of gender and survival)


print ("Percentage of Pclass=1 who survived:",train["Survived"][train["Pclass"]==1].value_counts(normalize=True)[1]*100)

print ("Percentage of Pclass=2 who survived:",train["Survived"][train["Pclass"]==2].value_counts(normalize=True)[1]*100)

print ("Percentage of Pclass=3 who survived:",train["Survived"][train["Pclass"]==3].value_counts(normalize=True)[1]*100)

print(0)
plt.show()

#4.C)SibSp feature
#--------------------------------
#draw a bar plot for SibSp vs.Survival
sbn.barplot(x="SibSp",y="Survived",data=train)
#I wont be printing individual percent values for ll of these
print ("Percentage of SibSp=0 who survived:",train["Survived"][train["SibSp"]==0].value_counts(normalize=True)[1]*100)

print ("Percentage of SibSp=1 who survived:",train["Survived"][train["SibSp"]==1].value_counts(normalize=True)[1]*100)

print ("Percentage of SibSp=2 who survived:",train["Survived"][train["SibSp"]==2].value_counts(normalize=True)[1]*100)

print()

plt.show()

#4.D)Parch Feature
#-------------------
#draw a bar plot for Parch vs survival
sbn.barplot(x="Parch",y="Survived",data=train)

plt.show()

#---------------------------------------------------------------
#people with less than four parents or children abroad are more likely to
#Again,people travelling alone are less likely to survive than those



#-------------------------------------------
#4.E)Age Feature
#-------------------------------------------

#sort the gaes into logical categories
train["Age"]=train["Age"].fillna(-0.5)    #filled the empty value of column where age missing value by -0.5(any negative number)
test["Age"]=test["Age"].fillna(-0.5)

bins=[-1,0,5,12,18,24,35,60,np.inf]       #np.inf:- positive infinity
labels=['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
train['AgeGroup']=pd.cut(train["Age"],bins,labels=labels)   #categorize the data and check data will categorize in which bins
test['AgeGroup']=pd.cut(test["Age"],bins,labels=labels)      #new column is created by use of feature engineering
print(train)
#draw a bar plot of Age vs. survival
sbn.barplot(x="AgeGroup",y="Survived",data=train)
plt.show()

#Done**********************************************************


#Some Observation from above output
#----------------------------------------------------
#Babies are more likely to survive than any other age group.

#--------------------
#4.F) Cabin Feature
#--------------------
#I tink the idea is that people with recorded cabin numbers are of
#and must more likely to survive.
train["CabinBool"]=(train["Cabin"].notnull().astype('int'))   #1 more column is added where value is not null answer will be true(204)
test["CabinBool"]=(test["Cabin"].notnull().astype('int'))      #true value will be represented as 1

print("############################\n\n")
print(train)

#calculate percentage of Cabin bool vs survived
print("Percentage ofCabinBool=1 who survived:",train["Survived"][train["CabinBool"]==1].value_counts(normalize=True)[1]*100)

print("Percentage ofCabinBool=0 who survived:",train["Survived"][train["CabinBool"]==0].value_counts(normalize=True)[1]*100)

#draw a bar plot of CabinBool vs.survival

sbn.barplot(x="CabinBool",y="Survived",data=train)

plt.show()

#5)Cleaning data
#*********************************************
#Time to clean our data to account for missing values and unnecesaary information
#Look at the Test Data
#Let's see how our test data looks

print (test.describe(include="all"))      #print also non numeric data
print (pd.isnull(test).sum())     #to check how many data missing in each column

#cabin feature
#We will start off by dropping the cabin feature since not a lot more useful
train=train.drop(['Cabin'],axis=1)    #whole column deleted
test=test.drop(['Cabin'],axis=1)

#Ticket fare
#We can also drop the Ticket feature since its unlikely to yield any useful
train=train.drop(['Ticket'],axis=1)
test=test.drop(['Ticket'],axis=1)


#Embarked Feature
#now we need to fillin the missing values in the Embarked feature
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

#replacing the missing value in Embarked feature with S

train["Embarked"]=train["Embarked"].fillna("S")
test["Embarked"]=test["Embarked"].fillna("S")


#Since a higher percenntage of values are missing it would be illogical to fill all values with same value
#create a combined group of both datasheets
combine=[train,test]               #if we want to run same code in more than one files
print(combine[0])         #to print first value

#extract a title for each Name in the train and test datsets
for dataset in combine:               #loop will run for two times
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    #every value of name their title is extracted
    #+ ->  is wild card expression which is used for previous expression which occur for any alphabet for one or more
    #* -> 0 or more              and /   -> any one character(may me integer also)
print ("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print (train)
print()

print(pd.crosstab(train['Title'],train['Sex']))    #which title have how manny males and females
#OUTPUT:-
#----------------
#Sex       female  male
#Title
#Capt           0     1
#Col            0     2
#Countess       1     0
#Don            0     1
#Dr             1     6
#Jonkheer       0     1
#Lady           1     0             Royal Family
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

#OUTPUT:
#Sex       female  male
#Title
#Master         0    40
#Miss         185     0
#Mr             0   517
#Mrs          126     0
#Rare           2    19
#Royal          1     1

print (train[['Title','Survived']].groupby(['Title'],as_index=False).count())     #will print only title and survived column and grouping is done with respect to Title

#OUTPUT:
#      Title  Survived
#0    Master        40
#1      Miss       185
#2        Mr       517
#3       Mrs       126
#4      Rare        21
#5     Royal         2

print ("\nMap each of the title groups to a numerical value")
title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Royal":5,"Rare":6}  #text column cannot be used so converting useful column to numeric
for dataset in combine:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)

print ("\nAfter replacing title with numeric value")
print (train)
print()

#OUTPUT:
'''
After replacing title with numeric value.

     PassengerId  Survived  Pclass  ...      AgeGroup CabinBool  Title
0              1         0       3  ...       Student         0    1.0
1              2         1       1  ...         Adult         1    3.0
2              3         1       3  ...    YoungAdult         0    2.0
3              4         1       1  ...    YoungAdult         1    3.0
4              5         0       3  ...    YoungAdult         0    1.0
5              6         0       3  ...       Unknown         0    1.0
6              7         0       1  ...         Adult         1    1.0
7              8         0       3  ...          Baby         0    4.0
8              9         1       3  ...    YoungAdult         0    3.0
9             10         1       2  ...      Teenager         0    3.0
10            11         1       3  ...          Baby         1    2.0
11            12         1       1  ...         Adult         1    2.0
12            13         0       3  ...       Student         0    1.0
13            14         0       3  ...         Adult         0    1.0
14            15         0       3  ...      Teenager         0    2.0
15            16         1       2  ...         Adult         0    3.0
16            17         0       3  ...          Baby         0    4.0
17            18         1       2  ...       Unknown         0    1.0
18            19         0       3  ...    YoungAdult         0    3.0
19            20         1       3  ...       Unknown         0    3.0
20            21         0       2  ...    YoungAdult         0    1.0
21            22         1       2  ...    YoungAdult         1    1.0
22            23         1       3  ...      Teenager         0    2.0
23            24         1       1  ...    YoungAdult         1    1.0
24            25         0       3  ...         Child         0    2.0
25            26         1       3  ...         Adult         0    3.0
26            27         0       3  ...       Unknown         0    1.0
27            28         0       1  ...       Student         1    1.0
28            29         1       3  ...       Unknown         0    2.0
29            30         0       3  ...       Unknown         0    1.0
..           ...       ...     ...  ...           ...       ...    ...
861          862         0       2  ...       Student         0    1.0
862          863         1       1  ...         Adult         1    3.0
863          864         0       3  ...       Unknown         0    2.0
864          865         0       2  ...       Student         0    1.0
865          866         1       2  ...         Adult         0    3.0
866          867         1       2  ...    YoungAdult         0    2.0
867          868         0       1  ...    YoungAdult         1    1.0
868          869         0       3  ...       Unknown         0    1.0
869          870         1       3  ...          Baby         0    4.0
870          871         0       3  ...    YoungAdult         0    1.0
871          872         1       1  ...         Adult         1    3.0
872          873         0       1  ...    YoungAdult         1    1.0
873          874         0       3  ...         Adult         0    1.0
874          875         1       2  ...    YoungAdult         0    3.0
875          876         1       3  ...      Teenager         0    2.0
876          877         0       3  ...       Student         0    1.0
877          878         0       3  ...       Student         0    1.0
878          879         0       3  ...       Unknown         0    1.0
879          880         1       1  ...         Adult         1    3.0
880          881         1       2  ...    YoungAdult         0    3.0
881          882         0       3  ...    YoungAdult         0    1.0
882          883         0       3  ...       Student         0    2.0
883          884         0       2  ...    YoungAdult         0    1.0
884          885         0       3  ...    YoungAdult         0    1.0
885          886         0       3  ...         Adult         0    3.0
886          887         0       2  ...    YoungAdult         0    6.0
887          888         1       1  ...       Student         1    2.0
888          889         0       3  ...       Unknown         0    2.0
889          890         1       1  ...    YoungAdult         1    1.0
890          891         0       3  ...    YoungAdult         0    1.0
'''
# In unknown we find the title of the passenger and then we find the mode value of their age and then replace it with.

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

#OUTPUT:-
'''
mode() of mr_age: 0    YoungAdult
Name: AgeGroup, dtype: category
Categories (8, object): [Unknown < Baby < Child < Teenager < Student < YoungAdult < Adult < Senior]


mode() of miss_age: 0    Unknown
Name: AgeGroup, dtype: category
Categories (8, object): [Unknown < Baby < Child < Teenager < Student < YoungAdult < Adult < Senior]


mode() of mrs_age: 0    Adult
Name: AgeGroup, dtype: category
Categories (8, object): [Unknown < Baby < Child < Teenager < Student < YoungAdult < Adult < Senior]


mode() of master_age: 0    Baby
Name: AgeGroup, dtype: category
Categories (8, object): [Unknown < Baby < Child < Teenager < Student < YoungAdult < Adult < Senior]


mode() of royal_age: 0    YoungAdult
1         Adult
Name: AgeGroup, dtype: category
Categories (8, object): [Unknown < Baby < Child < Teenager < Student < YoungAdult < Adult < Senior]


mode() of rare_age: 0    Adult
Name: AgeGroup, dtype: category
Categories (8, object): [Unknown < Baby < Child < Teenager < Student < YoungAdult < Adult < Senior]
'''


print ("\n ************************************* \n")
print ("\n *******   train[Age Group][0]: \n")

for x in range(10):                      #from 1 to 10 we want to print
    print (train["AgeGroup"][x])

age_title_mapping={1:"Young Adult",2:"Student",3:"Adult",4:"Baby",5:"Adult",6:"Adult"}

for x in range(len(train["AgeGroup"])):                         #loop works for 0 to 890
    if train["AgeGroup"][x]=="Unknown":                         #x=5(means for 6th record)
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]


for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x]=="Unknown":
        test["AgeGroup"][x]=age_title_mapping[test["Title"][x]]

print("\n After replacing Unknown values from Age Group column :\n")
print(train["AgeGroup"])

#OUTPUT:-
#After replacing Unknown values from Age Group column :
'''
     PassengerId  Survived  Pclass  ...      AgeGroup CabinBool  Title
0              1         0       3  ...       Student         0      1
1              2         1       1  ...         Adult         1      3
2              3         1       3  ...    YoungAdult         0      2
3              4         1       1  ...    YoungAdult         1      3
4              5         0       3  ...    YoungAdult         0      1
5              6         0       3  ...       Unknown         0      1
6              7         0       1  ...         Adult         1      1
7              8         0       3  ...          Baby         0      4
8              9         1       3  ...    YoungAdult         0      3
9             10         1       2  ...      Teenager         0      3
10            11         1       3  ...          Baby         1      2
11            12         1       1  ...         Adult         1      2
12            13         0       3  ...       Student         0      1
13            14         0       3  ...         Adult         0      1
14            15         0       3  ...      Teenager         0      2
15            16         1       2  ...         Adult         0      3
16            17         0       3  ...          Baby         0      4
17            18         1       2  ...       Unknown         0      1
18            19         0       3  ...    YoungAdult         0      3
19            20         1       3  ...       Unknown         0      3
20            21         0       2  ...    YoungAdult         0      1
21            22         1       2  ...    YoungAdult         1      1
22            23         1       3  ...      Teenager         0      2
23            24         1       1  ...    YoungAdult         1      1
24            25         0       3  ...         Child         0      2
25            26         1       3  ...         Adult         0      3
26            27         0       3  ...       Unknown         0      1
27            28         0       1  ...       Student         1      1
28            29         1       3  ...       Unknown         0      2
29            30         0       3  ...       Unknown         0      1
..           ...       ...     ...  ...           ...       ...    ...
861          862         0       2  ...       Student         0      1
862          863         1       1  ...         Adult         1      3
863          864         0       3  ...       Unknown         0      2
864          865         0       2  ...       Student         0      1
865          866         1       2  ...         Adult         0      3
866          867         1       2  ...    YoungAdult         0      2
867          868         0       1  ...    YoungAdult         1      1
868          869         0       3  ...       Unknown         0      1
869          870         1       3  ...          Baby         0      4
870          871         0       3  ...    YoungAdult         0      1
871          872         1       1  ...         Adult         1      3
872          873         0       1  ...    YoungAdult         1      1
873          874         0       3  ...         Adult         0      1
874          875         1       2  ...    YoungAdult         0      3
875          876         1       3  ...      Teenager         0      2
876          877         0       3  ...       Student         0      1
877          878         0       3  ...       Student         0      1
878          879         0       3  ...       Unknown         0      1
879          880         1       1  ...         Adult         1      3
880          881         1       2  ...    YoungAdult         0      3
881          882         0       3  ...    YoungAdult         0      1
882          883         0       3  ...       Student         0      2
883          884         0       2  ...    YoungAdult         0      1
884          885         0       3  ...    YoungAdult         0      1
885          886         0       3  ...         Adult         0      3
886          887         0       2  ...    YoungAdult         0      6
887          888         1       1  ...       Student         1      2
888          889         0       3  ...       Unknown         0      2
889          890         1       1  ...    YoungAdult         1      1
890          891         0       3  ...    YoungAdult         0      1
'''

age_mapping={"Baby":1,"Child":2,"Teenager":3,"Student":4,"Young Adult":5,"Adult":6,"Senior":7}
train["AgeGroup"]=train['AgeGroup'].map(age_mapping)
test["AgeGroup"]=test['AgeGroup'].map(age_mapping)

#dropping the agefeature for now,might change

train=train.drop(['Age'],axis=1)
test=test.drop(['Age'],axis=1)

print ("\n\nAge column dropped")
print (train)
#OUTPUT:
'''
Age column dropped.
     PassengerId  Survived  Pclass  ...   AgeGroup CabinBool  Title
0              1         0       3  ...        4.0         0      1
1              2         1       1  ...        6.0         1      3
2              3         1       3  ...        NaN         0      2
3              4         1       1  ...        NaN         1      3
4              5         0       3  ...        NaN         0      1
5              6         0       3  ...        NaN         0      1
6              7         0       1  ...        6.0         1      1
7              8         0       3  ...        1.0         0      4
8              9         1       3  ...        NaN         0      3
9             10         1       2  ...        3.0         0      3
10            11         1       3  ...        1.0         1      2
11            12         1       1  ...        6.0         1      2
12            13         0       3  ...        4.0         0      1
13            14         0       3  ...        6.0         0      1
14            15         0       3  ...        3.0         0      2
15            16         1       2  ...        6.0         0      3
16            17         0       3  ...        1.0         0      4
17            18         1       2  ...        NaN         0      1
18            19         0       3  ...        NaN         0      3
19            20         1       3  ...        NaN         0      3
20            21         0       2  ...        NaN         0      1
21            22         1       2  ...        NaN         1      1
22            23         1       3  ...        3.0         0      2
23            24         1       1  ...        NaN         1      1
24            25         0       3  ...        2.0         0      2
25            26         1       3  ...        6.0         0      3
26            27         0       3  ...        NaN         0      1
27            28         0       1  ...        4.0         1      1
28            29         1       3  ...        NaN         0      2
29            30         0       3  ...        NaN         0      1
..           ...       ...     ...  ...        ...       ...    ...
861          862         0       2  ...        4.0         0      1
862          863         1       1  ...        6.0         1      3
863          864         0       3  ...        NaN         0      2
864          865         0       2  ...        4.0         0      1
865          866         1       2  ...        6.0         0      3
866          867         1       2  ...        NaN         0      2
867          868         0       1  ...        NaN         1      1
868          869         0       3  ...        NaN         0      1
869          870         1       3  ...        1.0         0      4
870          871         0       3  ...        NaN         0      1
871          872         1       1  ...        6.0         1      3
872          873         0       1  ...        NaN         1      1
873          874         0       3  ...        6.0         0      1
874          875         1       2  ...        NaN         0      3
875          876         1       3  ...        3.0         0      2
876          877         0       3  ...        4.0         0      1
877          878         0       3  ...        4.0         0      1
878          879         0       3  ...        NaN         0      1
879          880         1       1  ...        6.0         1      3
880          881         1       2  ...        NaN         0      3
881          882         0       3  ...        NaN         0      1
882          883         0       3  ...        4.0         0      2
883          884         0       2  ...        NaN         0      1
884          885         0       3  ...        NaN         0      1
885          886         0       3  ...        6.0         0      3
886          887         0       2  ...        NaN         0      6
887          888         1       1  ...        4.0         1      2
888          889         0       3  ...        NaN         0      2
889          890         1       1  ...        NaN         1      1
890          891         0       3  ...        NaN         0      1
'''

#Name feature

#frop the name feature since it contains no more useful information.

train=train.drop(['Name'],axis=1)
test=test.drop(['Name'],axis=1)
print ("\n Name Dropped")
print (train)

#Sex feature
#map each sex value to a numerical value
sex_mapping={"male":0,"female":1}
train['Sex']=train['Sex'].map(sex_mapping)
test['Sex']=test['Sex'].map(sex_mapping)

print (train)
#OUTPUT:
'''
     PassengerId  Survived  Pclass  ...    AgeGroup  CabinBool  Title
0              1         0       3  ...         4.0          0      1
1              2         1       1  ...         6.0          1      3
2              3         1       3  ...         NaN          0      2
3              4         1       1  ...         NaN          1      3
4              5         0       3  ...         NaN          0      1
5              6         0       3  ...         NaN          0      1
6              7         0       1  ...         6.0          1      1
7              8         0       3  ...         1.0          0      4
8              9         1       3  ...         NaN          0      3
9             10         1       2  ...         3.0          0      3
10            11         1       3  ...         1.0          1      2
11            12         1       1  ...         6.0          1      2
12            13         0       3  ...         4.0          0      1
13            14         0       3  ...         6.0          0      1
14            15         0       3  ...         3.0          0      2
15            16         1       2  ...         6.0          0      3
16            17         0       3  ...         1.0          0      4
17            18         1       2  ...         NaN          0      1
18            19         0       3  ...         NaN          0      3
19            20         1       3  ...         NaN          0      3
20            21         0       2  ...         NaN          0      1
21            22         1       2  ...         NaN          1      1
22            23         1       3  ...         3.0          0      2
23            24         1       1  ...         NaN          1      1
24            25         0       3  ...         2.0          0      2
25            26         1       3  ...         6.0          0      3
26            27         0       3  ...         NaN          0      1
27            28         0       1  ...         4.0          1      1
28            29         1       3  ...         NaN          0      2
29            30         0       3  ...         NaN          0      1
..           ...       ...     ...  ...         ...        ...    ...
861          862         0       2  ...         4.0          0      1
862          863         1       1  ...         6.0          1      3
863          864         0       3  ...         NaN          0      2
864          865         0       2  ...         4.0          0      1
865          866         1       2  ...         6.0          0      3
866          867         1       2  ...         NaN          0      2
867          868         0       1  ...         NaN          1      1
868          869         0       3  ...         NaN          0      1
869          870         1       3  ...         1.0          0      4
870          871         0       3  ...         NaN          0      1
871          872         1       1  ...         6.0          1      3
872          873         0       1  ...         NaN          1      1
873          874         0       3  ...         6.0          0      1
874          875         1       2  ...         NaN          0      3
875          876         1       3  ...         3.0          0      2
876          877         0       3  ...         4.0          0      1
877          878         0       3  ...         4.0          0      1
878          879         0       3  ...         NaN          0      1
879          880         1       1  ...         6.0          1      3
880          881         1       2  ...         NaN          0      3
881          882         0       3  ...         NaN          0      1
882          883         0       3  ...         4.0          0      2
883          884         0       2  ...         NaN          0      1
884          885         0       3  ...         NaN          0      1
885          886         0       3  ...         6.0          0      3
886          887         0       2  ...         NaN          0      6
887          888         1       1  ...         4.0          1      2
888          889         0       3  ...         NaN          0      2
889          890         1       1  ...         NaN          1      1
890          891         0       3  ...         NaN          0      1
'''

embarked_mapping={"S":1,"C":2,"Q":3}
train['Embarked']=train['Embarked'].map(embarked_mapping)
test['Embarked']=test['Embarked'].map(embarked_mapping)
print()

print(train.head())

#OUTPUT:
#PassengerId  Survived  Pclass  ...    AgeGroup  CabinBool  Title
#0            1         0       3  ...         4.0          0      1
#1            2         1       1  ...         6.0          1      3
#2            3         1       3  ...         NaN          0      2
#3            4         1       1  ...         NaN          1      3
#4            5         0       3  ...         NaN          0      1

#fare feature
#fill in missing fare value in test set based on mean fare
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):             #null for missing value
        pclass=test["Pclass"][x]  #Pclass=3
        test["Fare"][x]=round(train[train["Pclass"]==pclass]["Fare"].mean(),2)   #2 decimal point round off

#map Fare values into groups
train['FareBand']=pd.qcut(train['Fare'],4,labels=[1,2,3,4])
test['FareBand']=pd.qcut(test['Fare'],4,labels=[1,2,3,4])

#drop Fare values
train=train.drop(['Fare'],axis=1)
test=test.drop(['Fare'],axis=1)
#check train data
print ("\n\nFare column dropped\n")
print (train)


#OUTPUT:
'''
            24         1       1    ...             1      1         4
24            25         0       3    ...             0      2         3
25            26         1       3    ...             0      3         4
26            27         0       3    ...             0      1         1
27            28         0       1    ...             1      1         4
28            29         1       3    ...             0      2         1
29            30         0       3    ...             0      1         1
..           ...       ...     ...    ...           ...    ...       ...
861          862         0       2    ...             0      1         2
862          863         1       1    ...             1      3         3
863          864         0       3    ...             0      2         4
864          865         0       2    ...             0      1         2
865          866         1       2    ...             0      3         2
866          867         1       2    ...             0      2         2
867          868         0       1    ...             1      1         4
868          869         0       3    ...             0      1         2
869          870         1       3    ...             0      4         2
870          871         0       3    ...             0      1         1
871          872         1       1    ...             1      3         4
872          873         0       1    ...             1      1         1
873          874         0       3    ...             0      1         2
874          875         1       2    ...             0      3         3
875          876         1       3    ...             0      2         1
876          877         0       3    ...             0      1         2
877          878         0       3    ...             0      1         1
878          879         0       3    ...             0      1         1
879          880         1       1    ...             1      3         4
880          881         1       2    ...             0      3         3
881          882         0       3    ...             0      1         1
882          883         0       3    ...             0      2         2
883          884         0       2    ...             0      1         2
884          885         0       3    ...             0      1         1
885          886         0       3    ...             0      3         3
886          887         0       2    ...             0      6         2
887          888         1       1    ...             1      2         3
888          889         0       3    ...             0      2         3
889          890         1       1    ...             1      1         3
890          891         0       3    ...             0      1         1
'''
#******************************************************
#6) Choosing the best model
#********************************************************

#Splitting the training data
#We will use part of our gtraining dtaa(20% in this case)to tewst the accuracy
#print(dataset.isnull().sum())
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

#OUTPUT:-
#Model-1: Accuracy of LogisticRegression: 77.09

#Model 2) Gaussian Bayes

from sklearn.naive_bayes import GaussianNB
gaussian=GaussianNB()
gaussian.fit(x_train,y_train)
y_pred=gaussian.predict(x_val)
acc_gaussian=round(accuracy_score(y_pred,y_val)*100,2)
print("Model-2: Accuracy of Logistic GaussianNB:",acc_gaussian)

#OUTPUT:-
#Model-2: Accuracy of Logistic GaussianNB: 77.65

#MODEL-3)Support Vector Machines
#---------------------------------
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_val)
acc_svc=round(accuracy_score(y_pred,y_val)*100,2)
print ("Model-3: Accuracy of Support Vector Machines:",acc_svc)

#OUTPUT:-
#Model-3: Accuracy of Support Vector Machines: 76.54

#MODEL-4) Linear SVC
from sklearn.svm import LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-4: Accuracy of LinearSVC : ",acc_linear_svc)

#OUTPUT:-
#MODEL-4: Accuracy of LinearSVC :  76.54

#MODEL-5) Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-5: Accuracy of Perceptron : ",acc_perceptron)

#OUTPUT:-
#MODEL-5: Accuracy of Perceptron :  45.81

#MODEL-6) Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-6: Accuracy of DecisionTreeClassifier : ", acc_decisiontree)

#OUTPUT:-
#MODEL-6: Accuracy of DecisionTreeClassifier :  72.63

#MODEL-7) Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-7: Accuracy of RandomForestClassifier : ",acc_randomforest)

#OUTPUT:-
#MODEL-7: Accuracy of RandomForestClassifier :  77.65

#MODEL-8) KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-8: Accuracy of k-Nearest Neighbors : ",acc_knn)

#OUTPUT:-
#MODEL-8: Accuracy of k-Nearest Neighbors :  75.98

#MODEL-9) Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-9: Accuracy of Stochastic Gradient Descent : ",acc_sgd)

#OUTPUT:-
#MODEL-9: Accuracy of Stochastic Gradient Descent :  74.3

#MODEL-10) Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print ("MODEL-10: Accuracy of GradientBoostingClassifier : ",acc_gbk)

#OUTPUT:-
#MODEL-10: Accuracy of GradientBoostingClassifier :  78.77

#Let us compare the accuracies of each model
models=pd.DataFrame({'Model':['Logistic Regression','Gaussian Naive Bayes','Support Vector Machines','Linear SVC','Perceptron',
                              'Decision Tree Classifier','Random Forest','KNN','Stochastic Gradient Descent','Gradient Boosting Classifier'],
                     'Score':[acc_logreg,acc_gaussian,acc_svc,acc_linear_svc,acc_perceptron,acc_decisiontree,
                              acc_randomforest,acc_knn,acc_sgd,acc_gbk]})
print()
print (models.sort_values(by='Score',ascending=False))

#OUTPUT:-
   '''
                 Model  Score
9  Gradient Boosting Classifier  78.77
1          Gaussian Naive Bayes  77.65
6                 Random Forest  77.65
0           Logistic Regression  77.09
2       Support Vector Machines  76.54
3                    Linear SVC  76.54
7                           KNN  75.98
8   Stochastic Gradient Descent  74.30
5      Decision Tree Classifier  72.63
4                    Perceptron  45.81
'''

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
