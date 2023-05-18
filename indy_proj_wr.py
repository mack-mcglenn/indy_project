#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import csv
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import seaborn as sb

from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')

# In[ ]:


def acq_nic():
    """ This function pulls my modified nicolas cage csv and converts it to a dataframe"""
    
    cage = pd.read_csv('nic_with_bo.csv')
    nic = pd.DataFrame(cage)
    
    return nic


# In[4]:


def assign_genre(df):
    """ This ridiculously long function classifies each of Mr. Cage's timeless works according to 
    genre."""
    for x in 'Movie':
        if 'Drive Angry' in 'Movie':
            return 'Action'
        elif 'Gone in 60 Seconds' in Movie:
            return 'Action'
        elif 'Dying of the Light' in Movie:
            return 'Action'
        elif 'Face/Off' in Movie:
            return 'Action'
        elif 'Ghost Rider' in Movie:
            return 'Action'
        elif 'Ghost Rider: Spirit of Vengeance' in Movie:
            return 'Action'
        elif 'The Rock' in Movie:
            return 'Action'
        elif "Captain Corelli's Mandolin" in Movie:
            return 'Action'
        elif "Snake Eyes" in Movie:
            return 'Action'
        elif 'Kiss of Death' in Movie:
            return 'Action'
        elif 'Deadfall' in Movie:
            return 'Action'
        elif 'Left Behind' in Movie:
            return 'Action'
        elif 'Gone in Sixty Seconds' in Movie:
            return 'Action'
        elif 'Jiu Jitsu' in Movie:
            return 'Action'
        elif 'Outcast' in Movie:
            return 'Action'
        elif 'Leaving Las Vegas' in Movie:
            return 'Drama'
        elif 'Raising Arizona' in Movie:
            return 'Drama'
        elif 'Adaptation' in Movie:
            return 'Drama'
        elif 'Birdy' in Movie:
            return 'Drama'
        elif 'Pig' in Movie:
            return 'Drama'
        elif 'Rumble Fish' in Movie:
            return 'Drama'
        elif 'Moonstruck' in Movie:
            return 'Drama'
        elif 'Bringing Out the Dead' in Movie:
            return 'Drama'
        elif 'Wild at Heart' in Movie:
            return 'Drama'
        elif 'Sonny' in Movie:
            return 'Drama'
        elif 'Never on Tuesday' in Movie:
            return 'Drama'
        elif 'World Trade Center' in Movie:
            return 'Drama'
        elif 'Zandalee' in Movie:
            return 'Drama'
        elif 'The Boy in Blue' in Movie:
            return 'Drama'
        elif 'Matchstick Men' in Movie:
            return 'Action'
        elif 'Kick-Ass' in Movie:
            return 'Action'
        elif 'Arsenal' in Movie:
            return 'Action'
        elif 'Lord of War' in Movie:
            return 'Drama'
        elif 'Inconceivable' in Movie:
            return 'Drama'
        elif 'City of Angels' in Movie:
            return 'Drama'
        elif 'Red Rock West' in Movie:
            return 'Drama'
        elif 'The Weather Man' in Movie:
            return 'Drama'
        elif 'Racing with the Moon' in Movie:
            return 'Drama'
        elif 'The Runner' in Movie:
            return 'Drama'
        elif 'Red Rock West' in Movie:
            return 'Thriller'
        elif 'Time to Kill' in Movie:
            return 'Thriller'
        elif '8MM' in Movie:
            return 'Thriller'
        elif 'The Trust' in Movie:
            return 'Thriller'
        elif 'The Frozen Ground' in Movie:
            return 'Thriller'
        elif 'Vengeance: A Love Story' in Movie:
            return 'Thriller'
        elif 'The Trust' in Movie:
            return 'Thriller'
        elif 'Seeking Justice' in Movie:
            return 'Thriller'
        elif 'Snake Eyes' in Movie:
            return 'Thriller'
        elif 'Stolen' in Movie:
            return 'Thriller'
        elif 'Trespass' in Movie:
            return 'Thriller'
        elif 'Windtalkers' in Movie:
            return 'Thriller'
        elif 'Grand Isle' in Movie:
            return 'Thriller'
        elif 'Primal' in Movie:
            return 'Thriller'
        elif 'Kill Chain' in Movie:
            return 'Thriller'
        elif 'Bangkok Dangerous' in Movie:
            return 'Action'
        elif 'Antigang' in Movie:
            return 'Thriller'
        elif 'Running with the Devil' in Movie:
            return 'Thriller'
        elif 'A Score to Settle' in Movie:
            return 'Thriller'
        elif 'Between Worlds' in Movie:
            return 'Thriller'
        elif 'Snowden' in Movie:
            return 'Thriller'
        elif 'Con Air' in Movie:
            return 'Thriller'
        elif 'Looking Glass' in Movie:
            return 'Thriller'
        elif 'Joe' in Movie:
            return 'Thriller'
        elif 'Dog Eat Dog' in Movie:
            return 'Thriller'
        elif 'Tokarev' in Movie:
            return 'Thriller'
        elif 'Next' in Movie:
            return 'Sci/Fi'
        elif 'Knowing' in Movie:
            return 'Sci/Fi'
        elif 'Color Out of Space' in Movie:
            return 'Sci/Fi'
        elif 'Pay the Ghost' in Movie:
            return 'Sci/Fi'
        elif 'Next' in Movie:
            return 'Sci/Fi'
        elif 'Season of the Witch' in Movie:
            return 'Sci/Fi'
        elif 'The Humanity Bureau' in Movie:
            return 'Sci/Fi'
        elif "Willy's Wonderland" in Movie:
            return 'Horror'
        elif 'Prisoners of the Ghostland' in Movie:
            return 'Horror'
        elif 'Grindhouse' in Movie:
            return 'Horror'
        elif "Mandy" in Movie:
            return 'Horror'
        elif 'Mom and Dad' in Movie:
            return 'Horror'
        elif 'Pay the Ghost' in Movie:
            return 'Horror'
        elif 'The Wicker Man' in Movie:
            return 'Horror'
        elif 'National Treasure' in Movie:
            return 'Family'
        elif 'National Treasure: Book of Secrets' in Movie:
            return 'Family'
        elif 'The Croods' in Movie:
            return 'Family'
        elif 'The Croods: A New Age' in Movie:
            return 'Family'
        elif "The Sorceror's Apprentice" in Movie:
            return 'Family'
        elif 'Spider-Man: Into the Spider-Verse' in Movie:
            return 'Family'
        elif 'Teen Titans Go! To the Movies' in Movie:
            return 'Family'
        elif 'Astro Boy' in Movie:
            return 'Family'
        elif 'G-Force' in Movie:
            return 'Family'
        elif 'The Ant Bully' in Movie:
            return 'Family'
        elif 'Christmas Carol: The Movie' in Movie:
            return 'Family'
        elif 'Moonstruck' in Movie:
            return 'RomCom'
        elif 'Trapped in Paradise' in Movie:
            return 'RomCom'
        elif 'It Could Happen to You' in Movie:
            return 'RomCom'
        elif 'Peggy Sue Got Married' in Movie:
            return 'RomCom'
        elif 'Honeymoon in Vegas' in Movie:
            return 'RomCom'
        elif 'Valley Girl' in Movie:
            return 'RomCom'
        elif 'City of Angels' in Movie:
            return 'RomCom'
        elif 'The Family Man' in Movie:
            return 'RomCom'
        elif 'Fast Times at Ridgemont High' in Movie:
            return 'RomCom'
        elif 'Windtalkers' in Movie:
            return 'War'
        elif 'USS Indianapolis: Men of Courage' in Movie:
            return 'War'
        elif 'Fire Birds' in Movie:
            return 'War'
        elif 'Industrial Symphony No. 1: The Dream of the Brokenhearted' in Movie:
            return 'Music'
        elif 'The Best of Times' in Movie:
            return 'Comedy'
        elif 'Amos and Andrew' in Movie:
            return 'Comedy'
        elif 'Guarding Tess' in Movie:
            return 'Comedy'
        elif "Vampire's Kiss" in Movie:
            return 'Comedy'
        elif 'Army of One' in Movie:
            return 'Comedy'
        elif 'Bad Lieutenant: Port of Call New Orleans' in Movie:
            return 'Crime'
        elif 'The Cotton Club' in Movie:
            return 'Crime'
        elif '211' in Movie:
            return 'Crime' 
        else:
            return 'Unknown'
        
        # Apply the function to create a new column in the dataframe
        nic['Genre'] = nic['Movie'].apply(assign_genre)
        return nic


# In[6]:


def prep_nic(nic):
    """ This function prepares my modified csv file for use in exploration and analysis.
    Specifically, it converts my Earnings and RT scores into numeric values and """
    
    ## ****UPDATES****
    # I'm changing the way to account for my categorical variables. Instead of making dummies, I'll use 'replace()' functions
    # to track genre and rating in modeling/analysis
    # These are my indexes which hold 'X' as a variable for RT score. I need to drop them before
    # I can convert the RT column to numeric
    nic['RottenTomatoes'] =nic['RottenTomatoes'].drop(index=[0, 13,14,82,92])
    
    # Converting to numeric
    nic['RottenTomatoes'] = pd.to_numeric(nic['RottenTomatoes'])
    nic['Earnings'] = pd.to_numeric(nic['Earnings'])
    
    # Get numeric values for rating & genre
    nic['Rating_num']= nic['Rating'].replace({'G':0, 'PG':1, 'PG-13':2, 'R':3, 'NR':4,'TV-NR':4, "TV-MA":3})
    nic['Genre_num']= nic['Genre'].replace({'Drama':1,'Thriller':2,'Action':3,'Family':4,'RomCom':5,'Horror':6,'Sci/Fi':7,'Comedy':8,
                     'Crime':0,'War':0,'Music':0})
    
    return nic


# In[5]:


def high_rt_chart(df):
    """ This function returns a histogram of Rotten Tomato scores over and under 80 measured against the decades long
        career of Nicolas Cage."""
    ax = plt.axes()
    ax.set_facecolor("mistyrose")
    ax.ticklabel_format(style='plain')
    plt.style.use('seaborn-deep')

    x =df[df.RottenTomatoes > 80].Year
    y = df[df.RottenTomatoes < 80].Year
    plt.yticks
    plt.hist([x, y], label=['RT Score > 80','RT Score < 80'])
    plt.legend(loc='upper left')

    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Count of Rotten Tomato Scores >/< 80 by Year')
    plt.show()


# In[7]:


def high_rt(nic):
    """ This function creates a new column that will return a 0 or 1 based on whether a film has a high Rotten Tomatoes score"""
    nic['High_RT'] = (nic.RottenTomatoes >=80).astype(int)
    
    return nic

def genre_test(nic):
    """This function will return the data for statistical analysis on drama vs thriller genres
    """
    a= 0.05
    t, p_value = stats.ttest_ind(nic[nic.Genre == 'Drama'].RottenTomatoes.dropna(), nic[nic.Genre == 'Thriller'].RottenTomatoes.dropna())
    print(f"T-statistic: {t:.3f}")
    print(f"P-value: {p_value:.3f}")
    if p_value/2 < a:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

def question_3(nic):

    # This will return all of the movies made after 2011 with RT scores over 80
    rog =  nic[(nic.Year >=2011) & (nic.RottenTomatoes >=80)]
    # This will return all of the movies recorded with RT scores over 80
    ham = nic[(nic.Year <2021) & (nic.RottenTomatoes >=80)]
    # Compare Variance
    print(f'Variance of high RT Scores for last decade: {rog.RottenTomatoes.var()}'),
    print(f'Variance of all time high RT Scores: {ham.RottenTomatoes.var()}')   

    # Compare # of Films
    print(f'Number of Films made after 2010 with RT scores of 80 and above: {rog.RottenTomatoes.count()}')
    print(f'Number of Films made since 1983 with RT scores of 80 and above: {ham.RottenTomatoes.count()}')

    # Compare High RT Scores
    print(f'Average High RT score for post-2010 films: {round(rog.RottenTomatoes.mean(),2)}')
    print(f'Average High RT score for all films: {round(ham.RottenTomatoes.mean(),2)}')

    # Compare All RT Scores
    print(f'Average Rotten Tomato Score for post-2010 films: {nic[nic.Year >2011].RottenTomatoes.mean()}')
    print(f'Average Rotten Tomato Score for all films: {nic[nic.Year >1982].RottenTomatoes.mean()}')
# In[8]:


def split_cage(df, target):
    '''
    take in a DataFrame return train, validate, test split on wine DataFrame.
    '''
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, val = train_test_split(train, test_size=.30, random_state=123, stratify=train[target])
    
    X_train, y_train = train.drop(columns=['Movie', 'Rating','Character', 'Year',
       'Earnings', 'Genre', 'High_RT']), train['High_RT']
    X_val, y_val = val.drop(columns=['Movie', 'Rating','Character', 'Year',
       'Earnings', 'Genre','High_RT']), val['High_RT']
    X_test, y_test = test.drop(columns=['Movie', 'Rating','Character', 'Year',
       'Earnings', 'Genre','High_RT']), test['High_RT']
    return X_train, X_val, X_test, y_train, y_val, y_test


def suggestion(df):
    df.head(3)
    df2 = df.drop(['Character','Voice','Year','Earnings','Rating_num', 'Genre_num'],axis=1)
    # df2= df2.set_index('Movie')
    df2['data'] = df2[df2.columns[1:]].apply(
        lambda x: ' '.join(x.dropna().astype(str)),
        axis=1)
    # print('df.data.head():')
    # print(df2['data'].head())

    # Vectorize DF
    vectorizer = CountVectorizer()
    vectorized = vectorizer.fit_transform(df2['data'])

    # Applying cosine similarity
    similarities = cosine_similarity(vectorized)
    # print(f'Array of cosine similarities:{similarities}')

    # Apply similarities to my df
    # Setting my movie titles as the index
    df = pd.DataFrame(similarities, columns=df['Movie'], index=df['Movie']).reset_index()
    # print(df.head())

    flick = input('What film did you enjoy?')
    recommendations = pd.DataFrame(df.nlargest(3,flick)['Movie'])
    recommendations = recommendations[recommendations['Movie']
                                    !=flick]
    print(f'''If you enjoyed "{flick}", you might enjoy: 
    {recommendations}''')



# In[ ]:




