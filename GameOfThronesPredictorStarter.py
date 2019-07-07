#!/usr/bin/env python
# coding: utf-8

"""
Scrape data and build a classifier for GOT survival
"""

#Import libs

#Data collection
import requests
from bs4 import BeautifulSoup

#Data processing
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

#Model building
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#Misc
import warnings
import sys
np.random.seed(seed=1)
warnings.filterwarnings('ignore')

print("Written in Python version 3.6")


#Data Collection

def getMovieCast():
    """
    This function downloads html from Rotten Tomatoes' Game of thrones page and extracts the full cast section items
    Input: None
    Output: BeatifulSoup Tag
    """

    print("Collecting movie cast list from rottentomatoes... ")
    GOTUrl = "https://www.rottentomatoes.com/tv/game-of-thrones"
    GOT_page = requests.get(GOTUrl)

    try:
        mainSoup = BeautifulSoup(GOT_page.content, 'html.parser')
        castSection = mainSoup.find(id="fullCast")
        castItems = castSection.find_all(class_="media-body")
        print(f"Full cast is {len(castItems)} actors and actresses")

        return castItems

    except:
        print (f"Could not access {GOTUrl} or it has been modified, cannot proceed without this data")


def parseMovieCast(movieCastTag):
    """
    This function parses movieCastTag to get following info:
     - their real name,
     - character name
     - url to actors/actresses page (this is passed to getCharacterIndivInfo() to parse out their information)
    The artists that are missing any of the info are excluded
    Input: BeatifulSoup Tag
    Output: List with actor/actress info
    """

    print("Parsing cast information (~1min)...")

    infoList = []
    try:
        for item in movieCastTag:
            findNameLink = item.find("a", class_="unstyled articleLink")

            if findNameLink:
                realName = findNameLink.span["title"]
                artistLink = findNameLink["href"]
            else:
                #print(f"Could not get name and link for {item} ")
                #Note on print: out of ~300 close to ~250 don't have this information
                #so not printing it here, todo: add this to a log
                continue

            findGOTName = item.find("span", class_="characters subtle smaller")
            if findGOTName:
                GOTName = findGOTName["title"]
            else:
                #print(f"Could not get character name for {item} ")
                continue

            mainUrl = "https://www.rottentomatoes.com"
            indivInfo = getCharacterIndivInfo(mainUrl + artistLink)
            if indivInfo:
                rating, seasons, bio = indivInfo
            else:
                rating, seasons, bio = np.nan

            infoList.append((realName, GOTName, rating, seasons, bio))

        return infoList
    except:
        print("Could not parse cast info")


def getCharacterIndivInfo(indivUrl):
    """
    This function takes URL for actor/actress Rotten Tomatoes page and returns Game of Thrones information from
    their filmography
    Input: URL string (e.g. "https://www.rottentomatoes.com/celebrity/lena_headey")
    Output: Tuple with Rating, appearance years and bip (e.g. (91, [2011, 2012], "..."))
    """

    indivPage = requests.get(indivUrl)

    try:
        indivSoup = BeautifulSoup(indivPage.content, 'html.parser')
        indivBioRaw = indivSoup.find('div', class_="col-sm-17 celeb_bio").get_text()
        indivBioList = [x.strip() for x in indivBioRaw.split('\n') if x.strip()]

        filmSection = indivSoup.find(id='filmography')
        tvSeries = filmSection.find(class_='table table-striped left')

        #Find GoT info from the list, skip first one - it is the header
        for row in tvSeries.find_all('tr')[1:]:
            if row["data-title"] == "Game of Thrones":
                seasonsNum = row["data-appearance-year"]
                return (row["data-rating"], row["data-appearance-year"], indivBioList)
    except:
        print(f"Data collection for {indivUrl} has failed")


#Data cleaning and feature engineering

def processData(actorInfoList):
    """
    This function reads actor/actress info into pandas dataframe and performs processing and feature engieneering
    Input: List with actor/actress information
    Output: Pandas dataframe - main processed data and dataframe linking index and actor name
    """
    try:
        df = pd.DataFrame(actorInfoList, columns=["realName","GOTName","rating","seasons","bio"])
        df.to_pickle("RawScrapedData.pickle")
        print(f"Selected cast is {df.shape[0]}")

        #RealName and GOTName are all unique IDs, copying to different table for referencing later
        dfIDs = df.loc[:,['realName','GOTName']]

        #'seasons' column will be converted to number of seasons which will become the learning target
        #initial 'seasons' will be dropped
        df['target_numSeasons'] = df['seasons'].apply(lambda s: len(s.strip("[").strip("]").split(',')) )

        #Family name matters quite a bit in the world of Game of thrones, creating a seperate column for family name
        #initial 'GOTName' will be dropped
        df['GOTName_Family']= df['GOTName'].apply(lambda s: s.split()[-1] if len(s.split()) > 1 else np.nan)

        #Label encode
        #initial 'GOTName_Family' will be dropped
        le = LabelEncoder()
        df['GOTName_Family_LE'] = le.fit_transform(df['GOTName_Family'].astype(str))

        #Bio column can be split to extract actor/actress rating and biography text
        #initial 'bio' column will be dropped
        df['HighestRating'] = df['bio'].apply(lambda s: int(s[1].strip('%'))/100)
        df['LowestRating'] = df['bio'].apply(lambda s: int(s[4].strip('%'))/100)
        df['bioText'] =  df['bio'].apply(lambda s: s[-1])

        #tf-idf on bioText and decomposing
        #Note: Personal biography is unlikely has any predicitive value for given task but will use it here for demo
        #initial 'bioText' will be dropped
        tfv = TfidfVectorizer(strip_accents='unicode',
                              analyzer='word',
                              token_pattern=r'(?u)\b\w+\b',
                              ngram_range=(1, 3),
                              use_idf=1,
                              smooth_idf=1,
                              sublinear_tf=1)
        svd = TruncatedSVD(n_components=8, random_state=1)

        tfidfCols = tfv.fit_transform(df['bioText'].values)
        svdCols = svd.fit_transform(tfidfCols)
        svdCols = pd.DataFrame(svdCols)
        svdCols = svdCols.add_prefix('TFIDF_')

        df = pd.concat([df, svdCols], axis=1)

        #Clean up
        #Drop 'rating' also as it is the same for all
        df.drop(['seasons','realName','GOTName', 'GOTName_Family','bio','bioText','rating'], axis=1, inplace=True)
        print("Final dataset (top rows):\n", df.head())

        return df, dfIDs
    except:
        print("Error processing data")


#Model building (baseline)

def buildModel(dataset):
    """
    This function trains and returns the model
    Input: Dataframe
    Output: XGBoost model
    """

    try:
        #Train a baseline model
        print("Building a prediction model")
        y_train = dataset['target_numSeasons']
        X_train = dataset.drop('target_numSeasons', axis = 1)

        #Build XGBoost model
        model = xgb.XGBClassifier(seed=1)
        model.fit(X_train, y_train, eval_metric='rmse')

        #Crossvalidation
        kfold = KFold(n_splits=5, random_state=1)
        result = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        print(f"Accuracy is {int(result.mean()*100)}%")

        return model

    except:
        print("Could not build the model")


def makePrediction(model, dataset, dfIDs):
    """
    This function takes trained model, data to make prediction and IDs
    """
    try:
        #Assuming we had a new character with following info:
        index = 0
        newCase = dataset.iloc[[index],1:]
        realName = dfIDs.loc[index, 'realName']
        GOTName = dfIDs.loc[index, 'GOTName']
        predicted_seasons = model.predict(newCase)

        print(f'Prediction: {realName} aka {GOTName} is going to play {int(predicted_seasons)} seasons of Game of Thrones')
    except:
        print("Could not make a prediction")


#Main

def main():

    try:
        movieCastTag = getMovieCast()
        actorInfoList = parseMovieCast(movieCastTag)
        dfProcessed, dfIDs = processData(actorInfoList)
        model = buildModel(dfProcessed)
        makePrediction(model, dfProcessed, dfIDs)
    except:
        print("Encountered an error")


if __name__ == "__main__":
    main()


# TODO:
# Good data:
# - Get the actual character description from Game of Thrones wiki:
# - Get images of actors/actresses and their played characters:
# - Use GOT Kaggle dataset
# Error handling
