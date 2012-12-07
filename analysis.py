import numpy as np
import csv
from sklearn import cross_validation
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def main():
    csv_file_object = csv.reader(open('train.csv', 'rb')) #Load in the training csv file
    header = csv_file_object.next() #Skip the fist line as it is a header
    train_data=[] #Create a variable called 'train_data'
    for row in csv_file_object: #Skip through each row in the csv file
        train_data.append(row) #adding each row to the data variable
    train_data = np.array(train_data) #Then convert from a list to an array
    #Normalize data
    train_data = train_data.astype(np.float)
    train_data = normalize(train_data)
    
    train, test = cross_validation.train_test_split(train_data, test_size=0.3, random_state=0)

    print 'Training'
    #In this case we'll use a random forest, but this could be any classifier
    forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    knn = KNeighborsClassifier()
    logit = LogisticRegression()
    svc = SVC()

    """
    print "Scoring"
    scores = cross_validation.cross_val_score(forest, train_data[0::,1::], train_data[0::,0])
    print "RF Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    scores = cross_validation.cross_val_score(knn, train_data[0::,1::], train_data[0::,0])
    print "KNN Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    scores = cross_validation.cross_val_score(logit, train_data[0::,1::], train_data[0::,0])
    print "Logit Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    print('LogisticRegression score: %f' %
    logit.fit(train[0::,1::], train[0::,0]).score(test[0::,1::], test[0::,0]))
    """
    print('KNN score: %f' %
        knn.fit(train[0::,1::], train[0::,0]).score(test[0::,1::], test[0::,0]))

    print('SVC score: %f' %
        svc.fit(train[0::,1::], train[0::,0]).score(test[0::,1::], test[0::,0]))
    """
    print('RandomForest score: %f' %
        forest.fit(train[0::,1::], train[0::,0]).score(test[0::,1::], test[0::,0]))
    """
if __name__=="__main__":
    main()