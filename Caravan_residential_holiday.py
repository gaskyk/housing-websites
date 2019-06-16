## Classifying caravan homes as residential or holiday homes

# Import packages
import numpy as np
import pandas as pd
from scipy.stats import sem
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2, RFECV
import operator

# Import park homes data collected from Zoopla API
# This is for PO, SO, NR and TR postcode areas
homes = pd.read_csv('../Zoopla/park_homes_type.csv')

print(homes.head())
print(homes.dtypes)
print(len(homes))

## Data wrangling

# Remove non-ASCII characters
def removeNonAscii(text):
    # Input is a list of strings
    text = [i.encode('ascii','ignore') for i in text]
    text = [i.decode("utf-8") for i in text]
    return text

# Remove punctuation
def removePunct(text):
    # Input is a list of strings
    from string import punctuation
    text = [''.join(c for c in s if c not in punctuation) for s in text]
    return text

# Combines functions to (1) convert text to lower case,
# (2) remove non-Ascii characters and (3) remove punctuation
def LowerAsciiPunc(text):
    # Input is a list of strings
    text = [i.lower() for i in text]
    text = removeNonAscii(text)
    text = removePunct(text)
    return text

# X = feature array
X = np.array(LowerAsciiPunc(homes['description']))

# y = label or target variable which you are trying to predict. Must be a number
# y=1 for holiday homes, y=0 for residential homes
y = np.where(homes['residential_holiday']=='H', 1, 0)
y = y.ravel()
print(sum(y), "park homes labelled as holiday homes out of", len(y), "(%0.1f per cent)" % ((sum(y))/len(y)*100))

## Feature (word) selection

# I haven't done cross validation for feature selection as my sample is already quite small (523 park homes)
# I think the feature selection is also already overfitting (lots of words don't seem useful to distinguish
# between a residential and holiday home)

# Train / Test split here
# Cross validation to a final evaluation of the goodness-of-fit of the chosen model with chosen features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
print("Shape of X_train:", X_train.shape)

# Look at all words to see which might have most signficance to problem

# Tf-idf (term frequency inverse document frequency) vectorising
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(stop_words=stopwords.words('english'))
X_train_select = vect.fit_transform(X_train)
print("Shape of X_train_select:", X_train_select.shape)

# Examine the k words / features with the highest chi-sq value when compared with y
def select_features(X, y, k1):
    # Inputs: X should be np.array, y should be of format np.array (but a vector), k1 should be an integer
    ch2 = SelectKBest(chi2, k = k1)
    ch2.fit_transform(X, y)
    # Print features and chi-sq scores of most relevance
    selected_names = (np.asarray(vect.get_feature_names())[ch2.get_support()]).tolist()
    scores = (np.around(np.asarray(ch2.scores_),4)[ch2.get_support()]).tolist()
    selected_features = dict(zip(selected_names, scores))
    print(sorted(selected_features.items(), key=operator.itemgetter(1), reverse=True))

select_features(X_train_select, y_train, 100)

# These are the words which I thought would be most useful for predicting between holiday and residential homes
# Many of the words above I thought would not be very useful for distinguishing between holiday and residential homes
words = ['holiday', 'swimming', 'family', 'owners', 'pool', 'beach', 'facilities', 'restaurant', 'entertainment',
         'boat', 'bar', 'holidays', 'countryside', 'weeks', 'tennis', 'families', 'club']

# For every word in words list, create a column in X_train (np.array): 1 if description contains word, 0 otherwise
X_df = pd.DataFrame(X_train)
X_df.columns = ['description']
for i in range(len(words)):
    X_df[words[i]] = np.where(X_df['description'].str.contains(words[i]), 1, 0)
X_train = np.array(X_df.iloc[:,1:])
print("Shape of X_train:", X_train.shape)

# Calculate Pearson correlation coefficient for presence of each word and whether a holiday home
from scipy.stats.stats import pearsonr
for i in range(len(words)):
    print(words[i])
    print("Correlation %0.3f, p-value %0.3f" % (pearsonr(X_train[:,i], y_train)))

## Machine learning

# Create a K-fold cross validation iterator with accuracy
# K is set to 10 - change if need be
def fit_evaluate_cv(model, X, y):
    model.fit(X, y)
    cv = StratifiedKFold(y, 10, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(scores)
    print("Mean accuracy: %0.3f (+/- %0.3f)" % (np.mean(scores), sem(scores)))

# Feature selection with 3-fold cross validation - 10-fold took a while
def select_features(model, X, y):
    feature_names = words
    rfecv = RFECV(model, step=1, cv=3)
    rfecv = rfecv.fit(X, y)
    print("Features sorted by their rank:")
    print(sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), feature_names)))

# Run model with features selected in select_features function
def fit_evaluate_cv_selected_features(model, X, y):
    rfecv = RFECV(model, step=1, cv=5, scoring='accuracy')
    rfecv = rfecv.fit(X, y)
    X_transformed = rfecv.transform(X)
    print("Shape of full training set: ",X.shape)
    print("Shape of reduced training set: ", X_transformed.shape)
    fit_evaluate_cv(model, X_transformed, y)

# Logistic regression
# L2 regularisation meant to better than L1 if you have less features than observations
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(penalty='l2')
fit_evaluate_cv(log, X_train, y_train)

# Decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='gini', max_depth=9)
fit_evaluate_cv(dtree, X_train, y_train)

# Random forests
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='gini')
fit_evaluate_cv(rf, X_train, y_train)

# Support Vector Machines grid search for optimisation
from sklearn import svm, grid_search
param_grid = [ {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1.0], 'kernel': ['linear','rbf']} ]
svmsearch = svm.SVC()
svmmodel = grid_search.GridSearchCV(svmsearch, param_grid, cv=3, scoring='accuracy')
svmmodel.fit(X_train, y_train)
print(svmmodel.best_params_)

# Create SVM with the best parameters
svmmodel = svm.SVC(C=10, gamma=0.1, kernel='rbf')
fit_evaluate_cv(svmmodel, X_train, y_train)

# TODO Apply final chosen model with selected features to remaining 20% testing set

