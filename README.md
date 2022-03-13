# JPMorganchasestockregressionanalysis


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm


data = pd.read_excel('C:/Users/Manav/Downloads/data.xlsx' )
print(data.describe())
plt.hist(data[['gold','oil','JPM']])

col1=data['Close_ETF']
col2=data['gold']

data.tail()

# Part-3


#fig1=plt.hist(data['Close_ETF'], edgecolor="black")
#def hist_plot()

sns.distplot(data['Close_ETF'], bins=10)
plt.xlabel('Sample')
plt.ylabel('ETF')
plt.title('ETF Histogram')
plt.grid(True)
#plt.ylim([0,0.05])
#plt.xlim([0,200])

plt.show()

sns.distplot(data['oil'], bins=10)
plt.xlabel('Sample')
plt.ylabel('oil')
plt.title('oil Histogram')
plt.grid(True)

sns.distplot(data['gold'], bins=10)
plt.xlabel('Count')
plt.ylabel('Gold')
plt.title('Gold Histogram')
plt.grid(True)

sns.distplot(data['gold'], bins=10)
plt.xlabel('Count')
plt.ylabel('Gold')
plt.title('Gold Histogram')
plt.grid(True)

Hypotheses

ETF
Null Hypothesis- H0- The distribution of the data is normal

Alternate Hypothesis- H1- The distribution of the data is not normal

OIL

Null Hypothesis- H0- The distribution of the data is normal

Alternate Hypothesis- H1- The distribution of the data is not normal


Gold

Null Hypothesis- H0- The distribution of the data is normal

Alternate Hypothesis- H1- The distribution of the data is not normal


JPM

Null Hypothesis- H0- The distribution of the data is normal

Alternate Hypothesis- H1- The distribution of the data is not normal

from scipy.stats import shapiro
from scipy.stats import anderson
from scipy.stats import kstest

#ShapiroWilk test
stat, p= shapiro(data['Close_ETF'])
print('Statistics=%.3f, p=%.3f' % (stat, p))

#interpretation
alpha=0.05

if p>alpha:
  print('Fail to reject H0')
else:
    print('Reject H0')

#AndersonDarling test

anderson(data['Close_ETF'], dist='norm')

#SmirnovKolmogorov test

kstest(data['Close_ETF'],'norm')

import statsmodels.api as sm
import pylab as py

#QQplot
sm.qqplot(data['Close_ETF'], line ='45')
py.show()

Normality test for OIL

#Shapiro Wilk test
stat, p= shapiro(data['oil'])
oil_SWtest=print('Statistics=%.3f, p=%.3f' % (stat, p))

#AndersonDarling test

anderson(data['oil'], dist='norm')

#SmirnovKolmogorov test

kstest(data['oil'],'norm')

#QQplot
sm.qqplot(data['oil'], line ='45')
py.show()

Normality test for GOLD

#Shapiro Wilk test
stat, p= shapiro(data['gold'])
oil_SWtest=print('Statistics=%.3f, p=%.3f' % (stat, p))

#AndersonDarling test

anderson(data['gold'], dist='norm')

#SmirnovKolmogorov test

kstest(data['gold'],'norm')

#QQplot
sm.qqplot(data['gold'], line ='45')
py.show()

Normality test for JPM

#Shapiro Wilk test
stat, p= shapiro(data['JPM'])
oil_SWtest=print('Statistics=%.3f, p=%.3f' % (stat, p))

#AndersonDarling test

anderson(data['JPM'], dist='norm')

#SmirnovKolmogorov test

kstest(data['JPM'],'norm')

#QQplot
sm.qqplot(data['JPM'], line ='45')
py.show()

Part-4

#sample calling

d1=df[['Close_ETF',]]

i2=[]
j2=[]
n2=[]

for seq in range(0,1000,20):
    i2.append(seq)
    
        
for seq1 in range(20,1020,20):
    j2.append(seq1)
        

    
for k in range(0,50):
    nn=d1[i2[k]:j2[k]]
    
    n2.append(nn)
    
    
print(n2)

#for calculating sample mean for 50 groups
mean=[]
for i in range(len(n2)):

    samplemean = np.mean(n2[i])
    mean.append(samplemean)
    
    print("#####################")
    
    print("Mean for sample number:",i+1)
    
    print(samplemean)

#for calculating standard deviation for 50 groups
stand=[]
for i in range(len(n2)):

    std = np.std(n2[i])
    stand.append(std)
    
    print("#####################")
    
    print(" Standard deviation for sample number:",i+1)
    
    print(std)

sns.distplot(mean)

sns.distplot(stand)

Part-5


# Creating a population replace with your own: 
population = data.Close_ETF.tolist()

sampleSize=10
value=100

for x in range(sampleSize):
    # Creating a random sample of the population with size 50: 
    sample = random.sample(population,value)  # With Replacment means the sample can contain the duplicates of the original population.
  

arr1 = np.array(sample)
arr1

import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return  m,m-h, m+h






Part-5 (2)

95% confidence interval for one of the10 simple random samples

mean_confidence_interval(sample, confidence=0.95)

# to get 50 simple random sample


sampleSize=50
value=20

for x in range(sampleSize):
    # Creating a random sample of the population with size 50: 
    sample2 = random.sample(population,value)

sample2

95% confidence interval for one of the 50 simple random samples

mean_confidence_interval(sample2, confidence=0.95)

From part-1 we can see that the population mean is 121.152960. The confidence interval for one of the 10 simple random samples (118.55831271106874, 124.21068708893125) and for one of the 50 simple random samples (115.21115748491609, 126.00284151508392) . Hence, the two intervals from (1) and (2) include the true value of the population mean 𝝁 which is 121.152960 as it lies within the confidence interval.
The 2nd case includes 50 simple random samples of the population, which is more than 10 simple random samples, and hence, a better representation of the populaion. So, it will be more accurate.
Since the true value lies within the interval for both the samples, we can say that both are accuarate in this case.

Part-6 

(T-test)
(1) & (2)

import scipy.stats as stats
stats.ttest_ind(data['Close_ETF'],sample)

stats.ttest_ind(data['Close_ETF'],sample2)

(3) & (4)

stats.f_oneway(sample,sample2)

stats.f_oneway(sample,data['Close_ETF'])

stats.f_oneway(sample2,data['Close_ETF'])

part-8

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

data=data.iloc[:999]

X=np.array(data['gold'])
X = X.reshape(-1,1)
y=np.array(data['Close_ETF'])
y = y.reshape(-1,1)

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)

model=LinearRegression()

scores = cross_val_score(model, X_train, y_train, cv= 5)
print(scores)

model=model.fit(X_train, y_train)
y_pred=model.predict(X_test)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(data['gold'], data['Close_ETF'])
ax.set_xlabel('input - x')
ax.set_ylabel('target - y')
plt.show()

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,model.predict(X_train),color='blue')

MSE = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error: ', MSE)
print('R2 Score: ', r2)



Removing outliers from gold and ETF

sns.boxplot(x=data['Close_ETF'])

sns.boxplot(x=data['gold'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(data['Close_ETF'], data['gold'])
ax.set_xlabel('Close_ETF')
ax.set_ylabel('Full-value gold')
plt.show()

z = np.abs(stats.zscore(data['gold']))
print(z)

data['gold'] = np.where(data['gold'] >.02, .001030,data['gold'])

data['gold'] = np.where(data['gold'] <-.015, .001030,data['gold'])


sns.boxplot(x=data['gold'])

fig2, ax = plt.subplots(figsize=(16,8))
ax.scatter(data['Close_ETF'], data['gold'])
ax.set_xlabel('Close_ETF')
ax.set_ylabel('Full-value gold')
plt.show()

data['Close_ETF'] = np.where(data['Close_ETF'] >129, 121.152960,data['Close_ETF'])

data['Close_ETF'] = np.where(data['Close_ETF'] <112, 121.152960,data['Close_ETF'])

data=data.iloc[:999]

X=np.array(data['gold'])
X = X.reshape(-1,1)
y=np.array(data['Close_ETF'])
y = y.reshape(-1,1)

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)

model=LinearRegression()

scores = cross_val_score(model, X_train, y_train, cv= 5)
print(scores)

model=model.fit(X_train, y_train)
y_pred=model.predict(X_test)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(data['gold'], data['Close_ETF'])
ax.set_xlabel('input - x')
ax.set_ylabel('target - y')
plt.show()

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,model.predict(X_train),color='blue')

MSE = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error: ', MSE)
print('R2 Score: ', r2)

minvalueIndexLabel = data['Close_ETF'].idxmin()
  
minvalueIndexLabel

Removing outliers from Close_ETF


data['Close_ETF'] = np.where(data['Close_ETF'] >129, 121.152960,data['Close_ETF'])

data['Close_ETF'] = np.where(data['Close_ETF'] <112, 121.152960,data['Close_ETF'])


part 8
(7)


99% confidence interval of the mean daily ETF return,

mean_confidence_interval(col1, confidence=0.99)

mean_confidence_interval(col2, confidence=0.99)

99% prediction interval of the individual daily ETF return.

mean_confidence_interval(y_pred, confidence=0.99)

mean_confidence_interval(X_test, confidence=0.99)

import pandas as pd
df = pd.read_excel('C:/Users/shiva/OneDrive/Desktop/DATA\dataExcel.xlsx')

print (df)


import matplotlib.pylab as plt
#plt.scatter(df.oil, df.gold)
df.plot(kind='scatter', x='oil', y='Close_ETF')

import matplotlib.pylab as plt
#plt.scatter(df.oil, df.gold)
df.plot(kind='scatter', x='gold', y='Close_ETF')

import matplotlib.pylab as plt
#plt.scatter(df.oil, df.gold)
df.plot(kind='scatter', x='JPM', y='Close_ETF')

import pandas as pd
df = pd.read_excel('C:/Users/shiva/OneDrive/Desktop/DATA\dataExcel.xlsx')

print (df)

print("Mean of population x: "+str(df.Close_ETF.mean()))
print("Std of population x: "+str(df.Close_ETF.std()))


import statistics
import matplotlib.pyplot as plt
import numpy as np

indexSize =df.count()/100
x=0
y=100
histogram={}

for i in range(0,10): #To iterate 10 samples
    apprix_1 = df.iloc[x:y:] #splitting into 100 values
    x+=100;
    y+=100
    print(str(i)+"th mean is :"+str(apprix_1.Close_ETF.mean()))
    histogram[i]=apprix_1.Close_ETF.mean()
    
#print(list(histogram.values())) 
plt.hist(list(histogram.values()))
print(list(histogram.values()))
print("\n Mean of Sample means:"+str(statistics.mean(histogram.values())))
print("\n Median of Sample means:"+str(statistics.median(histogram.values())))
print("\n Mode of Sample means:"+str(statistics.mode(histogram.values())))
print("\n Standard deviation of sample means:"+str(statistics.stdev(histogram.values())))



import statistics
import matplotlib.pyplot as plt
import random

# Creating a population replace with your own: 
population = df.Close_ETF.tolist()

sampleSize=10
value=100
histogram={};

for x in range(sampleSize):
    # Creating a random sample of the population with size 10: 
    sample = random.sample(population,value)  # With Replacment means the sample can contain the duplicates of the original population.
    #print("Mean:"+ str(statistics.mean(sample)))
    #print("Standard Deviation:"+ str(statistics.stdev(sample)))
    histogram[x]=statistics.mean(sample)
    
    
#print(list(histogram.values())) 
plt.hist(list(histogram.values()))
print(list(histogram.values()))
print("\n Mean of Sample means:"+str(statistics.mean(histogram.values())))
print("\n Median of Sample means:"+str(statistics.median(histogram.values())))
print("\n Mode of Sample means:"+str(statistics.mode(histogram.values())))
print("\n Standard deviation of sample means:"+str(statistics.stdev(histogram.values())))    

import numpy as np
import pandas as pd
from scipy import stats
import statistics

df = pd.read_excel('C:/Users/shiva/OneDrive/Desktop/DATA\dataExcel.xlsx')

#define F-test function
def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic 
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    p = 1-scipy.stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    return f, p

#perform F-test
fVal,pVal=f_test(df.oil, df.gold)
print(statistics.variance(df.oil))
print(statistics.variance(df.gold))
print("F value :"+str(fVal));
print("p value :"+str(pVal));
if((pVal)<0.05):
    print("Null hypothesis is rejected")
else:
    print("Null hypothesis is accepted")

import numpy as np

x = [18, 19, 22, 25, 27, 28, 41, 45, 51, 55,14, 15, 15, 17, 18, 22, 25, 25, 27, 34,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,19, 22, 25, 27, 28, 41, 45, 51, 55,14, 15, 15, 17, 18, 22, 25, 25, 27, 34,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,18, 19, 22, 25, 27, 28, 41, 45, 51, 55]
y = [14, 15, 15, 17, 18, 22, 25, 25, 27, 34,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,14, 15, 15, 17, 18, 22, 25, 25, 27, 34,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,14, 15, 15, 17, 18, 22, 25, 25, 27, 34,18, 19, 22, 25, 27, 28, 41, 45, 51, 55,14, 15, 15, 17, 18, 22, 25, 25, 27, 34,18, 19, 22, 25, 27, 28, 41, 45, 51, 55]

#define F-test function
def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic 
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    p = 1-scipy.stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    return f, p

#perform F-test
fVal,pVal=f_test(x, y)
print("F value :"+str(fVal));
print("p value :"+str(pVal));
print(statistics.variance(x))
print(statistics.variance(y))

import numpy as np
import pandas as pd
from scipy import stats
import statistics

df = pd.read_excel('C:/Users/shiva/OneDrive/Desktop/DATA\dataExcel.xlsx')
alpha=0.05
H0="Standard deviations are same"
Ha="Standard deviations are diffrent"

#define F-test function
def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.std(x, ddof=1)/np.std(y, ddof=1) #calculate F test statistic 
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    p = 1-scipy.stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    return f, p

#perform F-test
fVal,pVal=f_test(df.oil, df.gold)
print("Standard deviation of Oil::"+str(statistics.stdev(df.oil)))
print("Standard deviation of Gold::"+str(statistics.stdev(df.gold)))
if(statistics.stdev(df.oil)==statistics.stdev(df.gold)):
    print(H0);
else:
    print(Ha);
    
    
print("F value :"+str(fVal));
print("p value :"+str(pVal));
if((pVal)<alpha):
    print("Null hypothesis is rejected")
else:
    print("Null hypothesis is accepted")

import statistics as s
import matplotlib.pyplot as plt
import random
from scipy import stats
from statsmodels.stats import weightstats as stests
import numpy as np

# Creating a population replace with your own: 
goldData = df.gold.tolist()
oildata=df.oil.tolist()
apprix_1 = df.iloc[0:100:] 

value=10
alpha=0.05

goldMean=s.mean(goldData);
oilMean=s.mean(oildata);
#print("Gold's mean is ::"+str(goldMean));
#print("Oil's mean is::"+str(oilMean));
ttest,pval = stests.ztest(apprix_1.gold,apprix_1.oil)
print("p-value>>",pval)
if pval < alpha:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")


