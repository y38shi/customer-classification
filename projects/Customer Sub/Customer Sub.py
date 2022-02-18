#!/usr/bin/env python
# coding: utf-8

# # Customer Subscribtion Predictive Model
# 
# **Background:**
# 
# The marketing manager at a major financial institution in Canada want to run a campaign for the new product: bank term deposit. We have access to past campaign results on a similar product and using these data we want to build a predictive model to find customers that is most likely to purchase the new product (i.e. finding which customers we should target in our upcoming campaign).
# 
# **Goal:**
# 
# Our target is finding whether a customer will subscribe to the term deposit (y/n)

# _______________

# 
# 
# **Data Dictionary:**<br>
# 
# The data contains customer information collected from previous campaigns as well as the result from the last campaign. 
# 
# **Client related data:**<br>
# 1 - age (numeric)<br>
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')<br>
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)<br>
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')<br>
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')<br>
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')<br>
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')<br>
# 
# 
# **Related with the last contact:**<br>
# 8 - contact: contact communication type (categorical: 'cellular','telephone')<br>
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')<br>
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')<br>
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.<br>
# 
# 
# **Other attributes:**<br>
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)<br>
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)<br>
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)<br>
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')<br>
# 
# 
# **Social and economic context attributes:**<br>
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)<br>
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric)<br>
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)<br>
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)<br>
# 20 - nr.employed: number of employees - quarterly indicator (numeric)<br>
# 
# **Output variable (desired target):**<br>
# 21 - y - has the client subscribed? (binary: 'yes','no')<br>
# 
# 
# The dataset has 41,188 records and 21 columns, 19 of which can be considered for use in modelling

# _______

# **Data Exploration:**
# 
# Having a quick glance at the data, we are dealing with a pretty structured and complete dataset.
# 
# Immediately looking at the data, i know i want to break age down into groups. there could be other variables we need to change as well and we may discover them as we go.
# 
# We also have several several categorical variables where they are "unknown" labels. it will be worth looking at there variables to see the volume of "unkown" values.
# 
# I also have some intuition on what might affect a customer subbing or not, throughout my exploration, I will ask try to answer the following:
# 
# ***Hypothesis to test:***
# 
# 1. The number of time contacted does not affect outcome. in fact, conversion likely to taper off after a certain threshold because those that would have converted will have convertered already
# 
# 2. The outcome of previous campaign result affect outcome of future campaign. given campaign offer is selectly diff but within same space. i.e. those that convert for previou campaign is more likely to convert for future campaign.
# 
# **Summary of Finding:**
# 
# 1. Data Quality Issue
# - we have a couple volumns with outlier, but after investigation they don't appear to be bad data just odd. I have decided to keep these outliers
# - pdays seem to contain invalid values for when poutcome = 'failure'
# - on avg, <2% of the categorical data are "unknown"
# - however the variable default is very imbalanced with only 3 entry = 'yes'
# 
# |default|count|
# |-------|-----|
# |no|32588|
# |yes|3|
# |unknown|8597|
# 
# 2. Who subbed? (i.e. y='y')
# - Those contacted in a previous campaign are more likely to sub
#     - 88% of those **not** subbed were not contacted in a previous campaign, whereas only 68% of those subbed were not contacted in the previous campaign
# - In addition, those that subbed to previous campaign has even high chance of subbing
#     - 65% of those where their previous campaign outcome was success also subbed, whereas only 14% and 9% subbed for 'failure' and 'nonexistent' group respectively. this is quite a big difference (hypothesis 2)
# - the avg number of contacts for current campaign for those subbed is lower than those that did not sub
# - #contact does not increase sub %, in fact as # contact go up, sub % go down (hypothesis 1):
# <br>here we can see a clear downward trend
# ![title](img/NumContact.png)
# 
# 
# - those in the age group 25-34 were the most likely to sub, while those in age group 34-44 were the least likely to sub:
# ![title](img/Age.png)
# 
# 3. Other interesting finding
# - based on the dataset, we know the conversion rate for the previous campaign is ~11%
# - the job type with highest sub% are student and retired, lowest being blue-collar and entrepreneur
# - those with 'single' marital status have a bit higher sub %, butI suspect this is due to age instead of marital status
# - those with 'uni degree' have a bit higher sub %
# - loan data probably come from same source because those with housing loan = uknown, personal loan also = unknown

# _______

# **Feature Engineering:**
# 
# 1. Create age buckets (age_bucket)
# 2. Create pdays buckets, this will also deal with pdays = 999 (time_since_previous_contact)
# 3. Deal with "unknown" categorical variable using one hot encoding and remove "unknown" columns
# 4. Renaming of columns so it is more intuiative:<br>
#    campaign -> num_contact_current<br>
#    housing -> housing_loan <br>
#    loan -> personal_loan <br>
#    contact -> contact_method<br>
#    poutcome -> outcome_previous<br>
# 

# _______

# **Model Selection:**
# 
# What we know so far:
# - we want classify our data
# - the data we have is structured and labelled
# 
# Given the knowledge above, the first algorithm we can try is **Logistic Regression** and another one being  **Decision Trees**. As part of this exercise we will fit our data to both models and compare the results.
# 
# **Variable Selection:**
# 
# I will be excluding the following variables from the data set:
# 
# 1. duration cannot be used
# 2. default is imbalanced
# 3. age will be replaced with age_bucket
# 4. pdays will be replaced with time_since_previous_contact
# 
# ***optional:***<br>
# 1. marital not entirely valuable and we already have age, there was no indication that marital status affects outcome<br>
# 2. only pick one between job and education?  since one seems to affect another. Job seems to be a better candidate here

# ___

# **Model Comparison:**
# 
# |Metrics|Logistic Regression (all)|Logistic Regression (subset)|Decision Tree (all)|Decision Tree (subset)|
# |-------|-------------------|-------------|-------------------|-------------|
# |precision|0.67|0.65|0.46|0.49|
# |recall   |0.22|0.21|0.27|0.26|
# |fscore   |0.33|0.32|0.34|0.34|
# |accuracy |0.90|0.90|0.92|0.92|
# 
# 
# **Summary:**
# 
# 1. addition of informatiln did not improve results siginificantly
# 2. logistric regression has better precision while decision tree has better recall
# 3. accuracy about the same across the board
# 4. performance not ideal
# 
# What does this mean?
# 
# Our logistic regression model predicted 67% of all subs correctly, meaning that out of all the prediction our model made 67% of them actually subbed and 33% were predicted to have subbed but did not.
# 
# Meanwhile our decision tree model only predicted 49% of them correct but the recall is higher at 26%. This means that the decision tree model was able to predict subs more successfully than the logistic regression.
# 
# Putting this in more plain terms, imagine we had 100 subbed customer in our dataset. The decision tree model was able to identify 26 of these customers whereas the logistic regression was only able to identify 22. 
# _____________
# Note:
# subset includes the following features:
# 1. age bucket
# 2. num_contact_current
# 3. emp.var.rate
# 4. cons.price.indx,
# 5. cons.conf.indx,
# 6. euribor3m
# 7. nr.employed
# 8. job
# 9. martial
# 10. education
# 11. housing_loan
# 12. personal_loan
# 13. contact_method
# 14. outcome_previous
# 15. time_since_previous_contact
# _______________

# |	Features	|coefs	|abs	|+/-|
# |-------|------|------|----|
# |	month_may	|-0.644069|	0.644069	|negative|
# |	outcome_previous_success	|0.634225	|0.634225	|positive|
# |	outcome_previous_failure	|-0.530929	|0.530929	|negative|
# |	time_since_previous_contact_1 week ago|	0.467143|	0.467143	|positive|
# |	month_mar	|0.424676	|0.424676	|positive|
# |	time_since_previous_contact_2 weeks ago	|-0.367388	|0.367388|	negative|
# |	cons.price.idx	|0.336563	|0.336563	|positive|
# |	emp.var.rate|	-0.280221	|0.280221	|negative|
# |	month_jul	|0.270274	|0.270274	|positive|
# |	day_of_week_mon	|-0.244400|	0.244400	|negative|
# 
# <br> Logistic Regression, all features , top 10

# we can see the top 10 features here for our logistic regression. it picked up May as the top feature and it affects the result negatively. this could just be due to the fact that most of our data had last contact month as May and model incorrectly picked up the correlation of being contacted in May means less likely to sub. This is patterns we can derive from analyzing campaign data adn decide which month and day may be best to contact customers for higher success %. This is also why I regarded the month and day feature as part of my subset
# 
# A more obvious observation is that previous success postively affect result while previous failure negatively affect result. This is expected as we saw that people so who subbed before were more likely to sub again.

# |Features|	coefs	|abs|	+/-|
# |--------|----------|---|------|
# |	outcome_previous_success	|0.532748|	0.532748	|positive|
# |outcome_previous_failure	|-0.465285	|0.465285|	negative|
# |cons.price.idx	|0.433831|	0.433831	|positive|
# |	time_since_previous_contact_1 week ago	|0.423067	|0.423067	|positive|
# |contact_method_cellular	|0.383506	|0.383506	|positive|
# |contact_method_telephone	|-0.381523	|0.381523|	negative|
# |	time_since_previous_contact_2 weeks ago	|-0.363071	|0.363071	|negative|
# |	emp.var.rate	|-0.214587	|0.214587	|negative|
# |age_bucket_35-44	|-0.206783	|0.206783	|negative|
# |job_blue-collar	|-0.203925	|0.203925	|negative|
# 
# 
# <br> Logistic Regression, selected features, top 10
# 

# we can see here that for our subset. the top features are a lot more intuitive. and matches the pattern we discovered as part of our data analysis.
# 
# One interesting observation is the contact method, we can see that cellular postively affect the model while telephone negatively affect the model. Given that most people probably don't use landline phone these days, I am surprised by the difference in these features.

# |Features|	feature_importance|
# |---|---|
# |	nr.employed|	0.477480|
# |	outcome_previous_success|	0.086981|
# |	euribor3m|	0.084625|
# |	cons.conf.idx	|0.059969|
# |	num_contact_current	|0.037082|
# |	cons.price.idx	|0.035206|
# |	time_since_previous_contact_2 weeks ago|	0.017690|
# |	marital_single|	0.013101|
# |	job_retired	|0.012554|
# |	contact_method_telephone|	0.012431|
# 
# <br> Decision Tree, selected features , top 10

# For our tree model, the feature importance is a bit different. we can see here that the top feature is a lot more "important" than the subsequent features. It has also picked up a lot of the social and economic factors in the top 10

# |Features	|feature_importance|
# |---|---|
# |	nr.employed	|0.470225|
# |	outcome_previous_success|	0.084644|
# |	euribor3m|	0.074272|
# |	cons.conf.idx	|0.065584|
# |	num_contact_current|	0.033444|
# |	contact_method_telephone	|0.018325|
# |	time_since_previous_contact_2 weeks ago|	0.016695|
# |	day_of_week_mon	|0.013136|
# |	cons.price.idx	|0.010764|
# |	month_oct|	0.009825|
# 
# <br> Decision Tree, all features , top 10

# Almost no difference when we introduced more features

# ____

# #### Next Steps
# 
# Unfortunately none of the 2 models have high success in predicting subbed customers. In fact both model miss 70%-80% of the time. The purpose of this model is to inform agents on who to contact. If the model cannot capture the most likely to sub customers then it doesn't matter even if we were to predict 100% correct. This could cause our agent to miss out on the customers that would have subbed, although it can improve the success rate of our campaign.
# 
# However, improving our camapgin conversion % is not the goal of this and I don't believe either of these models are great at identifying customers that will sub to the bank product. This could be due to the fact that our data is high imbalanced (i.e. the number of customers subbed is significantly lower than the customer not subbed). With the original dataset, our model actually did a good job at predicting who will **not** sub but ultimately, this is not what we want.
# 
# what we can try next is by oversampling our data set so we can equal number of yes and no and that way our model can classify the customers more successfully.

# **Results after oversampling**
# 
# 
# |Metrics|Logistic Regression (all)|Logistic Regression (subset)|Decision Tree (all)|Decision Tree (subset)|
# |-------|-------------------|-------------|-------------------|-------------|
# |precision|0.97|0.92|0.90|0.89|
# |recall   |0.89|0.83|0.91|0.91|
# |fscore   |0.93|0.87|0.91|0.90|
# |accuracy |0.93|0.88|0.90|0.89|
# 
# 
# **Summary:**
# 
# Out model results are much much better! in fact this is a very good model. especially in our recall score, in our best model, we have a recall of 91%. meaning that we successfully identified 91% of all subbed customer.
# _____

# ____

# _______

# In[12]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy import stats, special
from sklearn import model_selection, metrics, linear_model, datasets, feature_selection, tree, preprocessing
import imblearn
pd.options.display.max_columns = 500
pd.set_option('display.max_rows',100)
pd.options.mode.chained_assignment = None


# In[2]:


full = pd.read_csv('full.csv')
full.head()


# In[3]:


full.shape


# In[4]:


full.columns.tolist()


# ### Data Exploration

# In[5]:


full.describe(include='all')


# **Quick Observations:**
# 
# - first glance, no missing values
# - age range from 17 - 98, so we've got quite a big range here. we may want to break this down into buckets
# - most common job is admin (~25%); contains unknown value
# - most customers are married (~60%); contains unknown value
# - most have uni degree (~30%); contains unknown value
# - most have no credit default (~79%); contains unknown value
# - most have housing loan (~52%); contains unknown value
# - most have no personal loan (~82%); contains unknown value
# - most have cell as contact method (~63%); ~37% have tele as contact method
# - May was the month where most customers were last contacted (~33%); possible indication that campaign launched in May or had most people working in May to call customers
# - Thursday was the day where most customer were last contacted; similar to above
# - Duration will not be used as part of modelling but we can see that standard deviation is quite high, we also have a max of 4918 -> which is almost 80mins
# - Clients on average contacted 2.5x during this campaign; but note that we do have a max value of 56. possible data error?
# - pdays is not a good value to look at here as it contain lots of 999 which indicates they've never been part of a previous campaign/never contacted for any previous campaign
# - previous range from 0-7, so we probably have lots of 0 values because a lot of customer were never contacted prior to this campaign; same as above, we can't draw much conclusion from just looking at this
# - poutcome, unlike pdays and previous. this is a bit more useful. from this we know 5,626 customers were contacted previously (~14%)
# - the economic and social variables don't mean much numerically
# 
# finally the output variable y, we know that 36,548 did not subscribe => meaning 4,640 customers subscribed, giving a 11% conversion rate for this campaign
# <br>
# <br>
# 

# In[6]:


sns.heatmap(full.corr().abs())


# Nothing too surprising here.. the social and economy variables have high correlation. which is expected given the type of data.
# 
# and.. because these variables are hard to describe using plain words. i'm actually going to exclude them for data exploration just to make things a little easier to read

# In[7]:


exclude_var = ['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
full_exp = full[full.columns[~full.columns.isin(exclude_var)]]


# Next i'm going to break down the categorical data to see if there's anything that stands out

# In[8]:


full_exp.groupby(['poutcome']).describe(include='all')


# **Quick Observation on poutcome:**
# - 4252 failure, 1373 success and 35563 not in previous campaign
# - the 3 groups pretty much follow the same pattern of the entire data set, with some exceptions
# - day_of_week for the 'failure' group is fri and not thursday
# - avg duration for the 'success' group is about 1 min higher than the other 2 groups; note here that sd is pretty high for all 3 groups. another interesting thing here is that we have very low durations of 1 sec.
# - pdays here is interesting.. because we have 999 value in the 'failure' group which shouldn't be possile -> possibly bad data
# - another interesting thing here is that those where poutcome = 'success' most of them have target variable y = 'y' (~65% of those that were a success target in the previous campain was also a success in current campain) **

# In[9]:


full_exp.groupby(['y']).describe(include='all')


# **Quick Observation on target variable:**
# - 4640 success and 36548 no conversion
# - those in the yes group is just a tad older
# - most common value for day_of_week to those in the 'no' group is mon 
# - duration highly affects the outcome, but here we can see that those in the 'yes' group has much higher avg duration; altho this could be because there are 0 values in the no group (aka those that didn't get contacted)
# - the avg number of contact in the yes group is lower than the no group; possible indicator that by contacting someone more will not drive conversion, but there's a outlier in the no group **
# - 88% in the no group was not in previous campaign, whereas 68% in the yes group was not in the previous campaign; this is likely good indication that customer that purchased similar product is more likely to buy again **
# - also nothing significant for the column contact, doesn't look to be a differentiator
# <br>
# 
# the relationship we notice bewteen poutcome and current target is interesting.. i will have a quick look at the breakdown between groups

# In[10]:


full_exp.groupby(['poutcome', 'y']).describe(include='all')


# **Observation:**
# - 14% subbed in 'failure' group
# - 9% subbed in 'nonexistent' group
# - 65% subbed in the 'success' group, this is a huge difference **
# - again.. those that subbed is just a tad older
# - same with number of contact, again.. lower in the yes group

# In[11]:


full_exp.groupby(['loan']).describe(include='all')


# **Observation:**
# - 2% of data is unknown, housing loan info is tied with this. probably same source of data (where loan = unknonw, housing = unknown) **
# - 11% of those in 'yes loan' group subbed and 12% of those in 'no loan' group subbed. so nothing interesting here

# In[12]:


full_exp.groupby(['housing']).describe(include='all')


# **Observation:**
# - like loan, 2% of data unknown
# - 9% of each group subbed. virtually no difference, same as loan above pretty much **

# In[13]:


full_exp.groupby(['default']).describe(include='all')


# **Observation**
# - so only 3 ppl had credit default.. this is a pretty inbalance column will likely exclude from modelling**

# In[14]:


full_exp.groupby(['job']).describe(include='all')


# **Observation:**
# - less than 1% data here is unknown
# - obvious age difference due to different job type (i.e. student and retired), same goes to marital status and education
# - unexpected data to see here is that majority of students have housing loan
# - 13% admin in yes group
# - 7% blue-collar in yes group
# - 9% entrepreneur in yes group
# - 10% housemaid in yes group
# - 12% housemaid in yes group
# - 26% retired in yes group
# - 11% self-employed in yes group
# - 18% services in yes group
# - 32% student in yes group
# - 11% technician in yes group
# - 14% unknown in yes group
# 
# our retired and student group has highest sub %, and blue-collar and entrepreneur have the lowest sub%. student group is suprising but I assume this is mature students given avg age is 26 **

# In[15]:


full_exp.groupby(['marital']).describe(include='all')


# **Observation:**
# - less than 1% data unknown
# - single people are younger as expected
# - 11% divorced in yes group
# - 11% married in yes group
# - 15% single in yes group
# 
# single group is slightly higher, but i suspect that this is due to age and not marital status **

# In[16]:


full_exp.groupby(['education']).describe(include='all')


# **Observation:**
# - 4% of data unknown
# - job and education closely tied, not surprised
# - only 18 in the illiterate group
# - uni degree had highest sub % (14%) vs rest of them. which hovers around 9-12% **

# In[17]:


full_exp.groupby(['contact']).describe(include='all')


# **Obsevation:** <br>
# -15% cell subbed and 6% of telephone subbed, i think this is mainly due to job type and not the contact method
# -will include this column regardless

# In[113]:


full_exp.groupby(['day_of_week']).describe(include='all')


# **Observation:**
# data is pretty evenly distributed

# In[114]:


full_exp.groupby(['month']).describe(include='all')


# Now I want to have a look at the numeric variables and see if i can find any relationship between them and target variable. we will do this by plotting most of them

# In[291]:


sns.set(rc = {'figure.figsize':(15,10)})


# In[19]:


sns.histplot(full_exp, x='age', hue='y',stat='count', bins = [15,25,35,45,55,65,100], kde=True)


# In[20]:


full_exp['age'].describe()


# In[21]:


#sns.catplot(data=full_exp, x='campaign', col='y',kind='count', height = 7, aspect=.9)
sns.histplot(full_exp[full_exp['campaign']<=10], x='campaign', hue='y',stat='count', discrete=True, multiple="stack")


# In[22]:


x=full_exp[full_exp['campaign']<=10].groupby(['campaign','y']).size()
x.groupby(level=0).apply(lambda x:  100 * x / float(x.sum()))


# **Observation:**
# general pattern here is as contact # increases sub% is also decreasing, confirming that # contact does not increase your success rate **

# In[23]:


sns.histplot(full_exp[full_exp['pdays']<999], x='pdays', hue='y',stat='count', multiple="stack",binwidth=7,kde=True)


# **Observation:**
# 
# majority of the customers were contacted within 14 days since last campaign (of those that were contacted)

# In[304]:


#job & sub 
sns.set(rc = {'figure.figsize':(15,10)})
sns.catplot(data=full_exp, x='job', col='y',kind="count").set_xticklabels(rotation=50)


# ### Data Cleaning

# First thing first.. some of the column names doesn't make sense intuitively to me, so to make thing easier to understand i will clean up the column names

# In[3]:


clean_data = full.copy()
clean_data.rename(columns={'housing':'housing_loan',
                          'loan':'personal_loan',
                          'contact':'contact_method',
                          'campaign':'num_contact_current',
                          'pdays':'days_since_previous',
                          'previous':'num_contact_previous',
                          'poutcome':'outcome_previous',},inplace = True)


# ___

# **Outliers**
# 
# During our data exploration we noticed that num_contact_current (campaign) and duration had very high max values. it could be bad data that should be excluded

# In[25]:


clean_data['duration'].plot()


# having a quick look at the plot, doesnt seem like the 4918sec conversation is bad data. at least nothing indicative here..
# 
# now doing the same for campaign column

# In[26]:


clean_data['num_contact_current'].plot()


# Again, 56 contacts is odd. but doesn't seem to be bad data since we have a couple records in 30s and 40s
# 
# **Outliers doesn't seem to be a problem for duration and campaign, so we will keep the data set as is**

# ___

# **Duplicated Values**
# 
# next we will see if there are duplicated values in the data set that should be removed

# In[4]:


len(clean_data[clean_data.duplicated()])


# seems like we have 12 duplicates. usually data set will have customer identifier so we know these are real duplicates and not just same value by chance. for the purpose of this project we are going to assume they are different customers and they are legitimate duplicated values

# In[5]:


clean_data.drop_duplicates(keep='first', inplace=True)


# ___

# **Incorrect Data**
# 
# During out exploration step, we noticed that when outcome_previous (poutcome) = 'failure, it still had pady = 999. this doesn't mean sense since 999 indicated that they've never been contacted before

# In[6]:


clean_data[clean_data['days_since_previous'] ==999].groupby(['outcome_previous'])['days_since_previous'].describe()


# This seems like there's quite a bit of 999. it most likely is incorrect/missing value
# 
# A couple ways we can tackle this:<br>
# - fill with NaN
# - fill with med
# - fill with mean
# - impute data
# 
# After consulting with the market managers they have indicated that we should just fill all these values with 14

# In[8]:


clean_data['days_since_previous'] = np.where((clean_data['outcome_previous'] == 'failure') & (clean_data['days_since_previous'] == 999),14,clean_data['days_since_previous'])

#sanity check
clean_data.groupby(['outcome_previous'])['days_since_previous'].describe()


# In[ ]:





# ___

# ### Data Processing
# 
# we will now change some of the column into a format that makes more sense based on our observation during data exploration

# 1. we will change the age into bucket. based on what we saw seems like 10 increment is a good start up until age 65

# In[13]:


#breaking up age into buckets

def age_bucket(row):
    if row['age'] <= 24:
        return '<24'
    elif (row['age'] >= 25 and row['age'] <= 34):
        return '25-34'
    elif (row['age'] >= 35 and row['age'] <= 44):
        return '35-44'
    elif (row['age'] >= 45 and row['age'] <= 54):
        return '45-54'
    elif (row['age'] >= 55 and row['age'] <= 64):
        return '55-64'
    elif row['age'] >= 65:
        return '65+'


# In[14]:


clean_data['age_bucket'] = clean_data.apply(age_bucket,axis=1)

#sanity check
clean_data.groupby(['age_bucket']).size()


# Now let's try the same plot with our new bucket

# In[33]:


sns.catplot(data=clean_data, x='age_bucket', col='y',kind="count", order= ['<24','25-34','35-44','45-54','55-65','65+'])


# this looks good, seems like we have a higher % of those in 25-34 in the yes group **

# 2. we will change days_since_previous also to something more meaningful

# In[15]:


def pdays_bucket(row):
    if row['days_since_previous'] <= 7:
        return '1 week ago'
    elif (row['days_since_previous'] >= 8 and row['days_since_previous'] <= 14):
        return '2 weeks ago'
    elif (row['days_since_previous'] >= 15 and row['days_since_previous'] <= 21):
        return '3 weeks ago'
    elif (row['days_since_previous'] >= 22 and row['days_since_previous'] <= 28):
        return '4 weeks ago'
    elif (row['days_since_previous'] >= 28 and row['days_since_previous'] < 999):
        return 'over 1 month ago'
    elif row['days_since_previous'] == 999:
        return 'never contacted'


# In[16]:


clean_data['time_since_previous_contact'] = clean_data.apply(pdays_bucket,axis=1)

#sanity check
clean_data.groupby(['time_since_previous_contact']).size()


# **Transforming columns**
# 
# Will be using one hot encoding to transform some of the categorical columns and will be dropping all where values is unknown

# In[102]:


clean_data.select_dtypes(include=['object']).columns.tolist()


# I'm going to exclude a few columns like month and day_of_week. while I believe when a campaign is ran and when a customer is contacted can be a factor in customer subbing or not. but i dont believe it is a good variable to explain customer behaviour and their likelihood to sub in the future

# In[17]:


#first create a copy with all variables
#remove duration because it is highly correlated
#also removing age and days_since_previous because we've already created columns for them
clean_data_allvar = clean_data.drop(columns = ['duration','age','days_since_previous'])

excl_col = ['duration','default','month','day_of_week','age','days_since_previous']
clean_data.drop(columns=excl_col, inplace=True)


# In[18]:


#reusing variable define above to get all categorical variable + whatever is missing
cat_cols = clean_data.select_dtypes(include=['object']).columns.tolist()
cat_cols_allvar = clean_data_allvar.select_dtypes(include=['object']).columns.tolist()
#removing our target variable
cat_cols.remove('y')
cat_cols_allvar.remove('y')


# In[19]:


clean_data = pd.get_dummies(clean_data, columns=cat_cols)
clean_data_allvar = pd.get_dummies(clean_data_allvar, columns=cat_cols_allvar)


# In[20]:


#removing columns that won't make sense in explaining our results
#we will keep all columns for the allvar version to compare results
excl_col_ohe = ['job_unknown','marital_unknown','education_unknown','housing_loan_unknown','personal_loan_unknown','education_illiterate']

clean_data.drop(columns=excl_col_ohe, inplace=True)


# In[169]:


clean_data_allvar.head()


# In[149]:


print(clean_data.shape)
print(clean_data_allvar.shape)


# ___

# ### Modelling

# First we will beign with logistic regression, we will start by creating the dataset for logistic regression

# In[27]:


#define target and indepent variables
X = clean_data[clean_data.columns.difference(['y'])]
y = clean_data['y']
X_all = clean_data_allvar[clean_data_allvar.columns.difference(['y'])]
y_all = clean_data_allvar['y']


# **Dealing with imbalanced data**
# 
# As we have observed previously, only around 11% of our data set contain y = 'yes', making the dataset quite imbalanced. We will be using SMOTE to combat this.
# 
# we will also be comparing the results of the original dataset vs the result after oversampling

# In[37]:


oversmaple = imblearn.over_sampling.SMOTE()
X_os, y_os = oversmaple.fit_resample(X,y)
X_os_all, y_os_all = oversmaple.fit_resample(X_all,y_all)
print(y.value_counts('yes'))
print(y_os.value_counts('yes'))


# We can see that after applying SMOTE, the new data set has equal number of yes and no. now we will move on to fitting our data to the logistic regression and decision tree models and compare the results

# In[39]:


#split into train and test set. stratified on y
train_X, test_X, train_y, test_y = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 3, stratify = y)
train_X_all , test_X_all , train_y_all , test_y_all  = model_selection.train_test_split(X_all , y_all , test_size = 0.3, random_state = 3, stratify = y)


# In[41]:


#split our oversampling data into train and test
train_X_os, test_X_os, train_y_os, test_y_os = model_selection.train_test_split(X_os, y_os, test_size = 0.3, random_state = 3)
train_X_os_all , test_X_os_all , train_y_os_all , test_y_os_all  = model_selection.train_test_split(X_os_all , y_os_all , test_size = 0.3, random_state = 3)


# In[42]:


print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)
print(train_X_all.shape)
print(train_y_all.shape)
print(test_X_all.shape)
print(test_y_all.shape)
print(train_X_os.shape)
print(train_y_os.shape)
print(test_X_os.shape)
print(test_y_os.shape)
print(train_X_os_all.shape)
print(train_y_os_all.shape)
print(test_X_os_all.shape)
print(test_y_os_all.shape)


# In[44]:


print(train_y.value_counts('yes'))
print(test_y.value_counts('yes'))
print(train_y_all.value_counts('yes'))
print(test_y_all.value_counts('yes'))
print(train_y_os.value_counts('yes'))
print(test_y_os.value_counts('yes'))
print(train_y_os_all.value_counts('yes'))
print(test_y_os_all.value_counts('yes'))


# In[48]:


#define model
model_lr = linear_model.LogisticRegression(solver='liblinear')
model_lr_all = linear_model.LogisticRegression(solver='liblinear')
model_lr_os = linear_model.LogisticRegression(solver='liblinear')
model_lr_os_all = linear_model.LogisticRegression(solver='liblinear')


# In[54]:


#fit to original data set
model_lr.fit(train_X,train_y)
model_lr_all.fit(train_X_all,train_y_all)
print(model_lr.score(train_X,train_y))
print(model_lr_all.score(train_X_all,train_y_all))
print(model_lr.score(test_X,test_y))
print(model_lr_all.score(test_X_all,test_y_all))


# In[55]:


#fit to oversample data set
model_lr_os.fit(train_X_os,train_y_os)
model_lr_os_all.fit(train_X_os_all,train_y_os_all)
print(model_lr_os.score(train_X_os,train_y_os))
print(model_lr_os_all.score(train_X_os_all,train_y_os_all))
print(model_lr_os.score(test_X_os,test_y_os))
print(model_lr_os_all.score(test_X_os_all,test_y_os_all))


# In[56]:


#coef on orginal data set of subset variables
coef_table1 = pd.DataFrame({"Features": list(train_X.columns)}).copy()
coef_table1.insert(len(coef_table1.columns),"coefs",model_lr.coef_.transpose())
coef_table1['abs'] = coef_table1['coefs'].abs()
coef_table1['+/-'] = np.where(coef_table1['coefs'] >0, 'positive','negative')
coef_table1.sort_values(by=['abs'],ascending = False).head(10)


# In[59]:


#coef on orginal data set of all variables
coef_table2 = pd.DataFrame({"Features": list(train_X_all.columns)}).copy()
coef_table2.insert(len(coef_table2.columns),"coefs",model_lr_all.coef_.transpose())
coef_table2['abs'] = coef_table2['coefs'].abs()
coef_table2['+/-'] = np.where(coef_table2['coefs'] >0, 'positive','negative')
coef_table2.sort_values(by=['abs'],ascending = False).head(10)


# In[102]:


#coef on oversample data set of subset variables
coef_table2 = pd.DataFrame({"Features": list(train_X_os.columns)}).copy()
coef_table2.insert(len(coef_table2.columns),"coefs",model_lr_os.coef_.transpose())
coef_table2['abs'] = coef_table2['coefs'].abs()
coef_table2['+/-'] = np.where(coef_table2['coefs'] >0, 'positive','negative')
coef_table2.sort_values(by=['abs'],ascending = False)


# In[58]:


#coef on oversample data set of all variables
coef_table2 = pd.DataFrame({"Features": list(train_X_os_all.columns)}).copy()
coef_table2.insert(len(coef_table2.columns),"coefs",model_lr_os_all.coef_.transpose())
coef_table2['abs'] = coef_table2['coefs'].abs()
coef_table2['+/-'] = np.where(coef_table2['coefs'] >0, 'positive','negative')
coef_table2.sort_values(by=['abs'],ascending = False).head(10)


# In[339]:


#validation of logistic regression on orginal data set with all variable
#for all variables
y_pred_lr = model_lr_all.predict(test_X_all)
y_actual_lr = test_y_all

#results = metrics.precision_recall_fscore_support(y_actual_lr, y_pred_lr, average = 'macro')
results = metrics.precision_recall_fscore_support(y_actual_lr, y_pred_lr, average = 'binary', pos_label = 'yes')
accuracy = metrics.accuracy_score(y_actual_lr,y_pred_lr)
precision = results[0]
recall = results[1]
fscore = results[2]

print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('fscore: ' + str(fscore))
print('accuracy: ' + str(accuracy))


# In[61]:


#validation of logistic regression on orginal data set with subset variable
y_pred_lr = model_lr.predict(test_X)
y_actual_lr = test_y

results = metrics.precision_recall_fscore_support(y_actual_lr, y_pred_lr, average = 'binary', pos_label = 'yes')
accuracy = metrics.accuracy_score(y_actual_lr,y_pred_lr)
precision = results[0]
recall = results[1]
fscore = results[2]

print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('fscore: ' + str(fscore))
print('accuracy: ' + str(accuracy))


# In[63]:


#validation of logistic regression on oversample data set with all variable
#for all variables
y_pred_lr = model_lr_os_all.predict(test_X_os_all)
y_actual_lr = test_y_os_all

#results = metrics.precision_recall_fscore_support(y_actual_lr, y_pred_lr, average = 'macro')
results = metrics.precision_recall_fscore_support(y_actual_lr, y_pred_lr, average = 'binary', pos_label = 'yes')
accuracy = metrics.accuracy_score(y_actual_lr,y_pred_lr)
precision = results[0]
recall = results[1]
fscore = results[2]

print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('fscore: ' + str(fscore))
print('accuracy: ' + str(accuracy))


# In[64]:


#validation of logistic regression on oversample data set with subset variable
#for all variables
y_pred_lr = model_lr_os.predict(test_X_os)
y_actual_lr = test_y_os

#results = metrics.precision_recall_fscore_support(y_actual_lr, y_pred_lr, average = 'macro')
results = metrics.precision_recall_fscore_support(y_actual_lr, y_pred_lr, average = 'binary', pos_label = 'yes')
accuracy = metrics.accuracy_score(y_actual_lr,y_pred_lr)
precision = results[0]
recall = results[1]
fscore = results[2]

print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('fscore: ' + str(fscore))
print('accuracy: ' + str(accuracy))


# **Decision Tree**

# In[96]:


model_dt = tree.DecisionTreeClassifier(max_depth=15)
model_dt_all = tree.DecisionTreeClassifier(max_depth=15)
model_dt_os = tree.DecisionTreeClassifier(max_depth=33)
model_dt_os_all = tree.DecisionTreeClassifier(max_depth=33)


# In[97]:


#fit original dataset
model_dt.fit(train_X,train_y)
model_dt_all.fit(train_X_all,train_y_all)
print(model_dt.score(train_X,train_y))
print(model_dt_all.score(train_X_all,train_y_all))
print(model_dt.score(test_X,test_y))
print(model_dt_all.score(test_X_all,test_y_all))


# In[86]:


#fit oversampling dataset
model_dt_os.fit(train_X_os,train_y_os)
model_dt_os_all.fit(train_X_os_all,train_y_os_all)
print(model_dt_os.score(train_X_os,train_y_os))
print(model_dt_os_all.score(train_X_os_all,train_y_os_all))
print(model_dt_os.score(test_X_os,test_y_os))
print(model_dt_os_all.score(test_X_os_all,test_y_os_all))


# In[87]:


#feature importance for decision tree on original data with subset variable
coef_table3 = pd.DataFrame({"Features": list(train_X.columns)}).copy()
coef_table3.insert(len(coef_table3.columns),"feature_importance",model_dt.feature_importances_.transpose())
coef_table3.sort_values(by=['feature_importance'],ascending = False).head(10)


# In[88]:


#feature importance for decision tree on original data with all variable
coef_table4 = pd.DataFrame({"Features": list(train_X_all.columns)}).copy()
coef_table4.insert(len(coef_table4.columns),"feature_importance",model_dt_all.feature_importances_.transpose())
coef_table4.sort_values(by=['feature_importance'],ascending = False).head(10)


# In[70]:


#feature importance for decision tree on oversample data with subset variable
coef_table3 = pd.DataFrame({"Features": list(train_X_os.columns)}).copy()
coef_table3.insert(len(coef_table3.columns),"feature_importance",model_dt_os.feature_importances_.transpose())
coef_table3.sort_values(by=['feature_importance'],ascending = False).head(10)


# In[90]:


#feature importance for decision tree on oversample data with all variable
coef_table4 = pd.DataFrame({"Features": list(train_X_os_all.columns)}).copy()
coef_table4.insert(len(coef_table4.columns),"feature_importance",model_dt_os_all.feature_importances_.transpose())
coef_table4.sort_values(by=['feature_importance'],ascending = False).head(10)


# In[99]:


#validation of decision tree on orginal data set with all variable
y_pred_dt = model_dt_all.predict(test_X_all)
y_actual_dt = test_y_all

results = metrics.precision_recall_fscore_support(y_actual_dt, y_pred_dt, average = 'binary', pos_label = 'yes')
accuracy = metrics.accuracy_score(y_actual_dt,y_pred_dt)
precision = results[0]
recall = results[1]
fscore = results[2]

print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('fscore: ' + str(fscore))
print('accuracy: ' + str(accuracy))


# In[92]:


#validation of decision tree on orginal data set with subset variable
y_pred_dt = model_dt.predict(test_X)
y_actual_dt = test_y

#results = metrics.precision_recall_fscore_support(y_actual_dt, y_pred_dt, average = 'macro')
results = metrics.precision_recall_fscore_support(y_actual_dt, y_pred_dt, average = 'binary', pos_label = 'yes')
accuracy = metrics.accuracy_score(y_actual_dt,y_pred_dt)
precision = results[0]
recall = results[1]
fscore = results[2]

print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('fscore: ' + str(fscore))
print('accuracy: ' + str(accuracy))


# In[93]:


#validation of decision tree on oversample data set with all variable
y_pred_dt = model_dt_os_all.predict(test_X_os_all)
y_actual_dt = test_y_os_all

results = metrics.precision_recall_fscore_support(y_actual_dt, y_pred_dt, average = 'binary', pos_label = 'yes')
accuracy = metrics.accuracy_score(y_actual_dt,y_pred_dt)
precision = results[0]
recall = results[1]
fscore = results[2]

print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('fscore: ' + str(fscore))
print('accuracy: ' + str(accuracy))


# In[94]:


#validation of decision tree on oversamle data set with subset variable
y_pred_dt = model_dt_os.predict(test_X_os)
y_actual_dt = test_y_os

results = metrics.precision_recall_fscore_support(y_actual_dt, y_pred_dt, average = 'binary', pos_label = 'yes')
accuracy = metrics.accuracy_score(y_actual_dt,y_pred_dt)
precision = results[0]
recall = results[1]
fscore = results[2]

print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('fscore: ' + str(fscore))
print('accuracy: ' + str(accuracy))


# In[78]:


#cross validation

# function for fitting trees of various depths on the training data using cross-validation
def run_cross_validation_on_trees(X, y, tree_depths, cv=10, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = tree.DecisionTreeClassifier(max_depth=depth)
        cv_scores = model_selection.cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores
  
# function for plotting cross-validation results
def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()

# fitting trees of depth 1 to 24
sm_tree_depths = range(1,40)
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(train_X_os_all, train_y_os_all, sm_tree_depths)

# plotting accuracy
plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, 
                               'Accuracy per decision tree depth on training data')


    
 


# In[79]:


idx_max = sm_cv_scores_mean.argmax()
sm_best_tree_depth = sm_tree_depths[idx_max]
sm_best_tree_cv_score = sm_cv_scores_mean[idx_max]
sm_best_tree_cv_score_std = sm_cv_scores_std[idx_max]
print('The depth-{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset'.format(
      sm_best_tree_depth, round(sm_best_tree_cv_score*100,5), round(sm_best_tree_cv_score_std*100, 5)))


# In[282]:


train_scores, test_scores = list(), list()
# define the tree depths to evaluate
values = [i for i in range(1, 30)]
# evaluate a decision tree for each depth
X_train=train_X
y_train=train_y

X_test =test_X
y_test = test_y


for i in values:
    # configure the model
    model = tree.DecisionTreeClassifier(max_depth=i)
    # fit model on the training dataset
    model.fit(X_train, y_train)
    # evaluate on the train dataset
    train_yhat = model.predict(X_train)
    train_acc = metrics.accuracy_score(y_train, train_yhat)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    test_yhat = model.predict(X_test)
    test_acc = metrics.accuracy_score(y_test, test_yhat)
    test_scores.append(test_acc)
    # summarize progress
    avg = (train_acc+test_acc)/2
    train_dff = avg-train_acc
    test_diff = avg-test_acc

    print('>%d, train: %.3f, test: %.3f, avg: %3f, diff: %3f, %3f' % (i, train_acc, test_acc,avg,train_dff,test_diff))
# plot of train and test scores vs tree depth
plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label='Test')
plt.legend()
plt.show()


# Next we will model using decision tree, similar to logistic regressino, we will define the dataset for decision tree

# Confusion Matrix

# 
# | |y|n|
# |-|-|-|
# |y|301 (TP)|151 (FP)|
# |n|1091 (FN)|10810 (TN)|
# logistic regression (all)
# 
# | |y|n|
# |-|-|-|
# |y|291 (TP)|159 (FP)|
# |n|1101 (FN)|10802 (TN)|
# <br>
# logistic regression (subset)
# 
# | |y|n|
# |-|-|-|
# |y|384 (TP)|294 (FP)|
# |n|1008 (FN)|10667 (TN)|
# decision tree (all)
# 
# | |y|n|
# |-|-|-|
# |y|365 (TP)|278 (FP)|
# |n|1027 (FN)|10683 (TN)|
# decision tree (subset)

# In[387]:


#ROC curve

metrics.plot_roc_curve(model_lr, test_X, test_y) 


# Try undersampling

# In[ ]:




