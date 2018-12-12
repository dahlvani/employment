import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

import math
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

file = r'downloads/Unemployment.xls'
df= pd.read_excel(file, header=None)
df.head(10)

#We need to clean the excel file for obvious visual defects

df.columns=df.iloc[7] #This renames the columns of the data frame
df=df.iloc[8:] #This deletes the data of rows with extraneous information
df=df.reset_index(drop=True) #This resets index without adding an additional column
df.head() # We can now explore the data to find additional items that need to be cleaned
df.shape
df.info()

# Most columns have most data. However, the data is stored as objects when in most columns floats would be more preferable
df=df.apply(pd.to_numeric,errors='ignore')
#Now we look for empty data.
#The above demonstrates that we have 3275 rows of data but some of our columns have as few as 3192 data points.
#Thus at most any column is missing as much as three percent of its data. Is this random or are there empty rows?

np.where(pd.isnull(df))
#This returns where the value is NaN (Not a number)

#Columns 50 and 51 are conspicuous in the amount of data they have missing as are rows 90 thru 99
#Columns 50 and 51 refer to income
#Let's find out what is going on with rows 90 thru 99
print(df.iloc[90:100,0:7])

#It seems like in particular, rows 92, 95, and 99 have their urban / rural codes missing.
#This could become a matter of importance depending on the analysis done.
#For now, I do not exclude anything from the data set.
#I suspect some vacancies are the result of statewide data. These rows should be excluded since most of the data is county level data.
print(df.iloc[102,2])

#I create a new data frame for US and state level data and remove the data from the data set I am utilizing

dfstate=df[df['Metro_2013'].isnull() & df['Urban_influence_code_2013'].isnull() & df['Rural_urban_continuum_code_2013'].isnull()]
df = df[np.isfinite(df['Metro_2013']) & np.isfinite(df['Rural_urban_continuum_code_2013'])& np.isfinite(df['Urban_influence_code_2013'])]
df.head()
df.shape

#Now that we have a relatively clean dataset, we can do some analysis
#We begin with data exploration and basic data visualization. We'll use matplotlib for exploratory data analysis
#Later, we'll use bokeh or seaborn for more complex charts to show our findings if we find anything

df['Median_Household_Income_2016'].describe()
dfstate['Median_Household_Income_2016'].describe()
df['Rural_urban_continuum_code_2013'].value_counts()

fig, axs = plt.subplots(1, 2,figsize=(10,5),sharey=False)
axs[0].hist(df['Median_Household_Income_2016'])
axs[1].hist(dfstate['Median_Household_Income_2016'])
fig.suptitle("Median Household Income Histogram")
fig.text(0.5, 0.04, 'Median Household Income 2016', ha='center', va='center')
fig.text(0.06, 0.5, 'Count', ha='center', va='center', rotation='vertical')
axs[0].set_title('By County')
axs[1].set_title('By State')
axs[0].tick_params(
    axis='both',          # changes apply to both x-axis and y-axis
    which='both',         # both major and minor ticks are affected
    bottom=False,         # ticks along the bottom edge are off
    top=False,
    left=False,
    right=False,          # ticks along the top edge are off
    labelbottom=True)     # labels along the bottom edge are off
axs[1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=True)
plt.show();


fig, axs = plt.subplots(1, 2,figsize=(10,5),sharey=True)
fig.suptitle("Income vs Unemployment")
axs[0].scatter(df.Median_Household_Income_2016,df.Unemployment_rate_2016)
axs[1].scatter(dfstate.Median_Household_Income_2016,dfstate.Unemployment_rate_2016)
fig.text(0.5, 0.04, 'Median Household Income 2016', ha='center', va='center')
axs[0].set_title('By County')
axs[1].set_title('By State')
fig.text(0.06, 0.5, 'Unemployment Rate 2016', ha='center', va='center', rotation='vertical');
axs[0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=True)
axs[1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=True)
plt.show();


plt.figure(figsize=(10,5))
plt.scatter(df.Rural_urban_continuum_code_2013,df.Unemployment_rate_2016)
plt.title('Rurality vs Unemployment')
plt.xlabel('Rurality Coding')
plt.ylabel('Unemployment')
plt.tick_params(axis='both',which='both',bottom=False, top=False, left=False, right=False, labelbottom=True)
plt.show();

fig,axs = plt.subplots(1,2,figsize=(10,5),sharey=False)
axs[0].scatter(df.Rural_urban_continuum_code_2013,df.Median_Household_Income_2016)
axs[1].hist(df['Rural_urban_continuum_code_2013'])
axs[0].set_title('Household Income vs Rurality')
axs[1].set_title('Rurality Histogram')
plt.tick_params(axis='both',which='both',bottom=False, top=False, left=False, right=False, labelbottom=True)
plt.show();

df['Unemployment_rate_2016']=1/(df['Unemployment_rate_2016']) #We transform the data to fit a multi-linear model

df['Median_Household_Income_2016'].corr(df['Unemployment_rate_2016']) #And see if any of our variables of interest are correlated with each other
df['Rural_urban_continuum_code_2013'].corr(df['Unemployment_rate_2016'])
df['Rural_urban_continuum_code_2013'].corr(df['Median_Household_Income_2016'])

#We now run a linear regression in order to predict median household income from the uncorrelated predictor variables rural urban continuum code and unemployment rate
#Normally, we would check to make sure that the conditions of a linear regression are met but we bypass this for simplicity and since there is little multicolinearity and there is a linear relationship between the predictors and income

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score

lin = df[np.isfinite(df['Unemployment_rate_2016']) & np.isfinite(df['Median_Household_Income_2016'])]
X = lin.loc[:,['Unemployment_rate_2016','Rural_urban_continuum_code_2013']]
target = lin.loc[:,['Median_Household_Income_2016']]

X_train, X_test, Y_train, Y_test = train_test_split(X,target, test_size=.2, random_state=5)
lm=LinearRegression()
lm.fit(X_train,Y_train)
pred_test = lm.predict(X_test)

print('Linear Regression Results')
print('Coefficients: ', lm.coef_)
print('Intercept:  ', lm.intercept_)
print('Mean Squared Error:  ', mean_squared_error(Y_test, pred_test))
print('Coefficient of Determination: %.2f' % r2_score(Y_test, pred_test));

from sklearn import tree

#For the Rural_urban_continuum_code_2013, the classification is as follows
#Metropolitan Counties*
    #Code	Description
    #1	Counties in metro areas of 1 million population or more
    #2	Counties in metro areas of 250,000 to 1 million population
    #3	Counties in metro areas of fewer than 250,000 population

#Nonmetropolitan Counties
    #4	Urban population of 20,000 or more, adjacent to a metro area
    #5	Urban population of 20,000 or more, not adjacent to a metro area
    #6	Urban population of 2,500 to 19,999, adjacent to a metro area
    #7	Urban population of 2,500 to 19,999, not adjacent to a metro area
    #8	Completely rural or less than 2,500 urban population, adjacent to a metro area
    #9	Completely rural or less than 2,500 urban population, not adjacent to a metro area

#We want to recode this so that we have a group of metropolitan counties, nonmetropolitan ccouties with urban populations of 2500 or more, and completely rural counties

df["Rural_urban_continuum_code_2013"] = df['Rural_urban_continuum_code_2013'].replace([1,2,3,4,5,6,7,8,9],[1,1,1,2,2,2,2,3,3])
df['Rural_urban_continuum_code_2013'].value_counts()

lin2 = df[np.isfinite(df['Unemployment_rate_2016']) & np.isfinite(df['Median_Household_Income_2016']) & np.isfinite(df['Rural_urban_continuum_code_2013']) &np.isfinite(df['Civilian_labor_force_2016'])]
X = lin2.loc[:,['Unemployment_rate_2016','Median_Household_Income_2016','Civilian_labor_force_2016']]
target = lin2.loc[:,['Rural_urban_continuum_code_2013']]

X_train, X_test, y_train, y_test = train_test_split(X, target, random_state=1)
model = tree.DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print('Decision Tree\nAccuracy Score:  ', accuracy_score(y_test,y_predict));

x_train, x_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=0)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train,y_train)
y_predict=logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)

predictions = logisticRegr.predict(x_test)
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 10);
