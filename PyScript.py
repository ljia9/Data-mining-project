# This is my work. -Leigh Jia

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor

def main():

    # Start things off by loading data
    print "Loading data..."
    raw_data = "datasets/z-training.txt"
    test_data = "datasets/z-test1.txt"
    dataset = np.loadtxt(raw_data, delimiter=",")
    test_X_set = np.loadtxt(test_data, delimiter=",")
    
    X = dataset[:,0:9]
    Y = dataset[:,9]

    #print X[0]
    #print test_X_set[0]
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = preprocessing.scale(X)
    Y_scaled = preprocessing.scale(Y)
    #print X_scaled[0]
    
    print "Training data..."
    print "\nScores:"
    print "-------"
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_scaled, Y_scaled, test_size=0.4, random_state=0)

    ''' 
    here use different regressors to estimate how strong they are. 
    first, the scaled versions of the datasets are used with cross validation sets to find the squared error and the validation score (accuracy)
    but the real predicted values are print below without scaling
    '''
    
    clf = RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0).fit(X, Y)
    clf_scale = RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0).fit(X_train, y_train)
    print "Random Forest score: ", clf_scale.score(X_test, y_test)
    print "error: ", mean_squared_error(y_test, clf_scale.predict(X_test))
    ans = clf.predict(test_X_set)
    #print ans
    #for val in ans:
    #    print val

    clf = GradientBoostingRegressor(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0).fit(X_train, y_train)
    print "Gradient Boosting score: ", clf.score(X_test, y_test)
    print "error: ", mean_squared_error(y_test, clf.predict(X_test))
    
    clf = AdaBoostRegressor(n_estimators=100).fit(X_train, y_train)
    print "AdaBoost score: ", clf.score(X_test, y_test)
    print "error: ", mean_squared_error(y_test, clf.predict(X_test))
    
    clf = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
    print "DecisionTree score: ", clf.score(X_test, y_test)
    print "error: ", mean_squared_error(y_test, clf.predict(X_test))
    
    clf = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
    print "KNeighbors score: ", clf.score(X_test, y_test)
    print "error: ", mean_squared_error(y_test, clf.predict(X_test))

    clf = SGDRegressor().fit(X_train, y_train)
    print "SGD score: ", clf.score(X_test, y_test)
    print "error: ", mean_squared_error(y_test, clf.predict(X_test))

    writePrediction(ans)
    #macroData()

######################
#  Helper Functions  #
######################
def macroData():
    raw_data = "macro_airport.txt"
    test_data = "new-data.txt"
    dataset = np.loadtxt(raw_data, delimiter=",")
    test_X_set = np.loadtxt(test_data, delimiter=",")
    X = dataset[:,0:9]
    Y = dataset[:,9]

    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = preprocessing.scale(X)
    Y_scaled = preprocessing.scale(Y)
    #print X_scaled[0]
    
    print "Training data..."
    print "\nScores:"
    print "-------"
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_scaled, Y_scaled, test_size=0.4, random_state=0)
    clf = RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0).fit(X, Y)
    clf_scale = RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0).fit(X_train, y_train)
    print "Random Forest score: ", clf_scale.score(X_test, y_test)
    print "error: ", mean_squared_error(y_test, clf_scale.predict(X_test))
    ans = clf.predict(test_X_set)

def writePrediction(predictions):
    '''
    Last part of the project, write the prediction model to a txt file for easy submission
    '''
    f = open("ans.csv", 'w')
    date = []
    count = 0 

    with open("format/dates1.txt") as d:
        for line in d:
            line = line.rstrip('\n')
            date.append(line)
    for val in predictions:
        f.write(str(date[count]))
        f.write(',')
        f.write(str(val))
        f.write('\n')
        count += 1

    f.close()

def loadData(fi):
    '''
    Load data and print out line numbers for error detection and editing dataset
    '''
    num = 1
    content = []
    with open(fi) as f:
        for line in f:
            print num
            line = line.rstrip('\n')
            row = line.split(',')
            float_row = np.array(map(float, row))
            content.append(float_row)
            num+=1

if __name__ == "__main__":
    main()
