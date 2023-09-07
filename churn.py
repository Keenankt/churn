## import package
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import gender_guesser.detector as gdg
from sklearn import preprocessing
from sklearn.model_selection import train_test_split    
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTENC


## read data
df = pd.read_csv("donor_data.csv", encoding='unicode_escape')
##
df.drop('row_id', axis=1, inplace=True)
## convert time to datetime object
def convert_time(raw_time):
    date_object = datetime.datetime.strptime(raw_time,"%d/%m/%Y")
    return date_object
df['donation_date'] = df['donation_date'].apply(convert_time)
## set current time for analysis
current_time = datetime.datetime(2018,12,30,0,0,0)
## mean donations
df_mean = df[['donor_id','donation']]
df_mean = df_mean.groupby('donor_id').mean()
## max donation
df_max = df[['donor_id','donation']]
df_max = df_max.groupby('donor_id').max()
## number of donations
df_count = df[['donor_id','donation']]
df_count = df_count.groupby('donor_id').count()
## first and last donation
df_donation_date = df[['donor_id','donation_date']]
df_first = df_donation_date.groupby('donor_id').min()
df_last = df_donation_date.groupby('donor_id').max()
## get latest mail pref & state
df_mail_pref = df[['donor_id','mail_pref','state']]
df_mail_pref = df_mail_pref.groupby('donor_id').max()
df_mail_pref = df_mail_pref[['mail_pref','state']]

## merge summary list
df_summary = df_mean.merge(df_max,left_index=True,right_index=True)
df_summary = df_summary.merge(df_count,left_index=True,right_index=True)
df_summary = df_summary.merge(df_first,left_index=True,right_index=True)
df_summary = df_summary.merge(df_last,left_index=True,right_index=True)
df_summary = df_summary.merge(df_mail_pref,left_index=True,right_index=True)
df_summary.columns = ['mean','max','count','first_donation','last_donation','mail_pref','state']

## calculated fields
# coerce timedelta object to float
def convert_to_days(time_delta_obj):
    return time_delta_obj/datetime.timedelta(days=1)
## days "active" donor
df_summary['days_active_td'] = (df_summary['last_donation'] - df_summary['first_donation'])
df_summary['days_active'] = df_summary['days_active_td'].apply(convert_to_days)
## days since last donation
df_summary['days_since_last_donation_td'] = (current_time - df_summary['last_donation'])
df_summary['days_since_last_donation'] = df_summary['days_since_last_donation_td'].apply(convert_to_days)
## donor currently active or lapsed
## if donated at least once in last 12 months == active
def status_classifier(last_donation):
    days_since_last_donation = (current_time - last_donation)
    if days_since_last_donation >= datetime.timedelta(365):
        status = "lapsed"
    else:
        status = "active"
    return status
df_summary['status'] = df_summary['last_donation'].apply(status_classifier)
## guess gender based on name
## split out name to guess based on first name
split_names = df['donor_name'].str.split(pat= ' ',expand=True)
df[['first','last', 'last2']] = split_names
df.drop({'donor_name','last','last2'},axis=1,inplace=True)
## create gender detector function
detector = gdg.Detector() 
def guess_gender(first):
    gender = detector.get_gender(first)
    return gender
## apply detector to frame
df['gender'] = df['first'].apply(guess_gender)
## map gender to donor_id
gender_dict = df[['donor_id','gender']]
gender_dict = gender_dict.drop_duplicates('donor_id')
gender_dict.set_index('donor_id',inplace=True)
df_summary = df_summary.merge(gender_dict,left_index=True,right_index=True)

## create dataframe of useful features
df_model = df_summary[['mean','max','count','mail_pref','state','gender','days_active','status']]
encoder = preprocessing.LabelEncoder()
df_model = df_model.apply(encoder.fit_transform)

## prepare training data
x = df_model.drop('status', axis=1)
y = df_model['status']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33,random_state=30)

## oversample lapsed donors to level class with active
oversampler = SMOTENC(categorical_features=[3,4,5],random_state=30)
x_train, y_train = oversampler.fit_resample(x_train,y_train)

## train model
trained_model = RandomForestClassifier(random_state=30)
trained_model.fit(x_train,y_train)

## test model
predictions = trained_model.predict(x_test)
print(accuracy_score(y_test,predictions))

## feature importance
plt.barh(trained_model.feature_names_in_,trained_model.feature_importances_)
plt.title('Feature Importance')
plt.show()

## drop unusused features and retrain model
x_train_trimmed = x_train.drop(['gender','state','mail_pref'], axis=1)
x_test_trimmed = x_test.drop(['gender','state','mail_pref'], axis=1)

trained_model = RandomForestClassifier(random_state=30)
trained_model.fit(x_train_trimmed,y_train)

## test model
predictions = trained_model.predict(x_test_trimmed)
print(accuracy_score(y_test,predictions))

plt.barh(trained_model.feature_names_in_,trained_model.feature_importances_)
plt.show()

## merge model data for visualisation
df_x = x_train.append(x_test)
df_y = y_train.append(y_test)
df_summary = df_x.merge(df_y,left_index=True,right_index=True)
## visualise distribution of data
## useful features
fig, axs = plt.subplots(2,2,sharex=True)
fig.suptitle('Distribution of Useful Features, Active vs Churned Donors')
sns.violinplot(df_summary,x='status',y='mean',ax=axs[0,0])
axs[0,0].set_title('Average Donation')
axs[0,0].set_ylabel('$ Dollars')
axs[0,0].set_xlabel('')
sns.violinplot(df_summary,x='status',y='max',ax=axs[0,1])
active_patch = mpatches.Patch(color='tab:blue', label='Active')
lapsed_patch = mpatches.Patch(color='tab:orange', label='Lapsed')
axs[0,1].legend(handles=[active_patch,lapsed_patch])
sns.move_legend(axs[0,1], 'upper left',bbox_to_anchor=(1,1))
axs[0,1].set_title('Maximum Donation')
axs[0,1].set_ylabel('$ Dollars')
axs[0,1].set_xlabel('')
sns.violinplot(df_summary,x='status',y='count',ax=axs[1,0])
axs[1,0].set_title('Number of Donations')
axs[1,0].set_ylabel('Count')
axs[1,0].set_xlabel('')
axs[1,0].set_xticklabels('')
sns.violinplot(df_summary,x='status',y='days_active',ax=axs[1,1])
axs[1,1].set_title('Lifetime as an Active Donor')
axs[1,1].set_ylabel('Days')
axs[1,1].set_xlabel('')
axs[1,1].set_xticklabels('')
# shared axis
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Status")

plt.show()



