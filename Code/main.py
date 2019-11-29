'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def neural_predict(bow_train, train_labels, bow_test, test_labels):
    
    clf = MLPClassifier(solver='adam', alpha=0.0001, 
                        hidden_layer_sizes=(100,20,2), random_state=0,
                        activation='relu', early_stopping=False,
                        learning_rate='constant', learning_rate_init=0.01)
    
    clf.fit(bow_train, train_labels) 
    
    test_predict = clf.predict(bow_test)
    
    train_predict = clf.predict(bow_train)
    
    print('Neural Network train accuracy = {}'.format((train_predict == train_labels).mean()))

    print('Neural Network test accuracy = {}'.format((test_predict == test_labels).mean()))
    
def decision_tree(bow_train, train_labels, bow_test, test_labels):
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(bow_train, train_labels) 
    
    test_predict = clf.predict(bow_test)
    
    train_predict = clf.predict(bow_train)
    
    print('Decision Tree train accuracy = {}'.format((train_predict == train_labels).mean()))

    print('Decision Tree test accuracy = {}'.format((test_predict == test_labels).mean()))
    
    
    
def Ada_Boost(bow_train, train_labels, bow_test, test_labels):
    
    clf = AdaBoostClassifier(n_estimators=500, learning_rate=1)
    
    clf.fit(bow_train, train_labels) 
    
    test_predict = clf.predict(bow_test)
    
    train_predict = clf.predict(bow_train)
    
    print('boost train accuracy = {}'.format((train_predict == train_labels).mean()))

    print('boost test accuracy = {}'.format((test_predict == test_labels).mean()))
    
    return clf

def Support_Vector_Machine(bow_train, train_labels, bow_test, test_labels):
    
    
    #clf = svm.SVC(C=1, decision_function_shape='ovo', gamma=1)
    
    
    #clf = svm.SVC(C=1, degree=10, kernel='poly')
    
    clf = svm.LinearSVC(C=0.1)
    
    clf.fit(bow_train, train_labels) 
    
    test_predict = clf.predict(bow_test)
    
    train_predict = clf.predict(bow_train)
    
    print('svm train accuracy = {}'.format((train_predict == train_labels).mean()))

    print('svm test accuracy = {}'.format((test_predict == test_labels).mean()))
    
    confusion = confusion_matrix(test_predict, test_labels)
    
    return clf, confusion

def Random_Forest(bow_train, train_labels, bow_test, test_labels):  
    
    clf = RandomForestClassifier()
    clf.fit(bow_train, train_labels) 
    
    test_predict = clf.predict(bow_test)
    
    train_predict = clf.predict(bow_train)
    
    print('random forest train accuracy = {}'.format((train_predict == train_labels).mean()))

    print('random forest test accuracy = {}'.format((test_predict == test_labels).mean()))
    
    return clf

def M_B(bow_train, train_labels, bow_test, test_labels):
    
    clf = MultinomialNB(alpha=0.01)
    clf.fit(bow_train, train_labels) 
    
    test_predict = clf.predict(bow_test)
    
    train_predict = clf.predict(bow_train)
    
    print('binomial train accuracy = {}'.format((train_predict == train_labels).mean()))

    print('binomial test accuracy = {}'.format((test_predict == test_labels).mean()))
    
    confusion = confusion_matrix(test_predict, test_labels)
    
    return clf
    

def confusion_matrix(predict, test):
    
    confusion = np.zeros([20,20])
    
    N = test.size
    
    for i in range(N):
        if predict[i] != test[i]:
            confusion[test[i]][predict[i]] += 1
            
    print (confusion)
    
    #plt.imshow(confusion,cmap='hot')
    #plt.show()
    
    return confusion

if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)

    #bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    
    train_w, test_w, w_names = tf_idf_features(train_data, test_data)
    
    
    #clf_neural = neural_predict(train_w, train_data.target, test_w, test_data.target)
    
    clf =  M_B(train_w, train_data.target, test_w, test_data.target)
    
    #decision_tree(train_w, train_data.target, test_w, test_data.target)

    #Random_Forest(train_w, train_data.target, test_w, test_data.target)
    
    #Ada_Boost(train_w, train_data.target, test_w, test_data.target)
    
    #clf, confusion = Support_Vector_Machine(train_w, train_data.target, test_w, test_data.target)
    
    
    
    #parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'C':[1, 10]}
    
    #parameters = {'learning_rate':[0.01, 0.001]}
    
    #parameters = {'hidden_layer_sizes':[100, 50, 20]}
     
    
    #parameters = {'n_estimators':[370], 'min_samples_leaf':[50,100,200,300]}
    
    #parameters = {'C':[0.5,0.8,0.3]}
    #parameters = {'gamma':[30,50,70]}
    
    #parameters = {'learning_rate':[0.01,0.1,1]}
    
    #parameters = {'n_estimators':[100,500,1000]}
    #svc = svm.SVC()
    
    #gbc = GradientBoostingClassifier()
    
    #DTC = tree.DecisionTreeClassifier()
    #RFC = RandomForestClassifier()
    
    #SVC = svm.LinearSVC()
    #ADA = AdaBoostClassifier(n_estimators=50)
    
    #SVC = svm.SVC(decision_function_shape='ovo', gamma=0.01)
    
    #GBC = GradientBoostingClassifier(learning_rate=0.01)
    
    #clf = GridSearchCV(ADA, parameters)
    #clf.fit(train_w, train_data.target)
    #train_predict = clf.predict(train_w)
    #test_predict  = clf.predict(test_w)
    
    #print('train accuracy = {}'.format((train_predict == train_data.target).mean()))

    #print('test accuracy = {}'.format((test_predict == test_data.target).mean()))
    
    
    
    
    
    
    
    
    