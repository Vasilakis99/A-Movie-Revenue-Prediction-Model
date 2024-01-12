#import needed libraries

import pandas as pd
import numpy as np
import seaborn as ans
import statsmodels.api as sm
import matplotlib.pyplot as plt


#read and print dataset

df = pd.read_csv(r'C:\Users\user\Desktop\\mov.csv')

# make and save the new binary variable explaining if a movie is english or not

df["English_language"] = df["original_language"].apply(lambda x: 1 if x == "en" else 0)
df.to_csv('English_language', index=False)

############### SECTION 1 #########################

#make new dataset with only wanted columns

mov_need = df[['budget','popularity','runtime','English_language','revenue']]


#find correlation for each variable

r1 = mov_need.revenue.corr(mov_need.budget)
r2 = mov_need.revenue.corr(mov_need.popularity)
r3 = mov_need.revenue.corr(mov_need.English_language)
r4 = mov_need.revenue.corr(mov_need.runtime)


#print r_i for i in {1,2,3,4}

print('Correlation Coeffitient for revenues and budget is:',r1)
print('Correlation Coeffitient for revenues and popularity is:',r2)
print('Correlation Coeffitient for revenues and english Language is:',r3)
print('Correlation Coeffitient for revenues and runtime is:',r4)


#make scatterplots

####### BUDGET ################
budget = mov_need["budget"]
revenue = mov_need["revenue"]

plt.scatter(budget, revenue)
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.title("Scatter Plot: Budget vs Revenue")
plt.scatter(budget, revenue, color='green')
plt.show()

print('r =',r1)

######### RUNTIME ##############

popularity = df["popularity"]
revenue = df["revenue"]

plt.scatter(popularity, revenue)
plt.xlabel("Popularity")
plt.ylabel("Revenue")
plt.title("Scatter Plot: Popularity vs Revenue")
plt.scatter(popularity, revenue, color='purple')
plt.show()

print('r =', r2)

######### POPULARITY #####################

English = df["English_language"]
revenue = df["revenue"]

plt.scatter(English, revenue)
plt.xlabel("English_language")
plt.ylabel("Revenue")
plt.title("Scatter Plot: Movies in English vs Revenue")
plt.scatter(English, revenue, color='purple')
plt.show()

print('r =',r3)

########## ENGLISH #############

runtime = df["runtime"]
revenue = df["revenue"]

plt.scatter(runtime, revenue)
plt.xlabel("Running Time")
plt.ylabel("Revenue")
plt.title("Scatter Plot: Running Time vs Revenue")
plt.scatter(runtime, revenue, color='purple')
plt.show()

print('r =', r4)


############### SECTION 2 #########################


# make and save the new binary variable explaining if a movie has budget larger than the average or not

mov_need.loc[:, "AVG_budget"] = mov_need["budget"].apply(lambda x: 1 if x >= 22531334.11 else 0)
mov_need.to_csv('AVG_budget', index=False)

mov_need = mov_need.drop('AVG BUDGET', axis=1)


#apply multiple reggresion

equation = sm.formula.ols(formula = 'revenue ~ budget + popularity  + English_language + runtime + AVG_budget ', data = mov_need)
multireg = equation.fit()
print(multireg.summary2())


############### SECTION 3 #########################

#checking conditions for multi reg 

X = mov_need[['budget','popularity','English_language','runtime','AVG_budget']]
Y = mov_need.revenue
predictions = multireg.predict(X)
res = Y - predictions
plt.hist(res)
plt.show()


plt.style.use('ggplot')
sns.pairplot(mov_need)
plt.show()
