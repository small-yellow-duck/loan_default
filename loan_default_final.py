# Kaggle loan default prediction 
# author: small yellow duck

import numpy as np  
import scipy as sp  
import pandas as pd  
import time
import random
import sklearn as sklearn


from sklearn import linear_model, svm
from sklearn import cross_validation
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier,RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


### usage
#run loan_default_final.py
#data = getData()
#data = convert(data)
#cols = getcols(data)
#X_train, Y_train, X_test, Y_test, imp = prepdata3(data, 18000, 4500, cols)	

def conv(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return np.nan 
	
def getData():  
	print "reading data using pandas"  
	#data = pd.read_table('train.csv', sep=',')  
	#print data  
	#print "converting data to numpy array"  
	#data = np.asarray(data)  
	#print data  
	start = time.clock()
	data = pd.read_csv("train_v2.csv") #, nrows=20000
	print 'Total clock time = ' + str(time.clock() - start)
	
	#data = pd.read_table('train.csv', sep=',')

	return data  
	
	
def convert(data):
	cols =list(data.columns.values)
	
	print 'converting objects to integers'
	for col in cols:
		if data[col].dtype == 'object':
			data[col] = data[col].apply(conv)
			
	
	print 'done converting'

	return data
	
	
def getcols(data):
	cols =list(data.columns.values)
	cols.remove('loss')
	cols.remove('id')
	return cols
	
#morecols = [0,269,270,2,10,619,64,401,403,756,758,590,592,220,660,620,250,331,373,375,379,658,22,621,378,329] 
#startercols = [520,521,271,1,268]
#colst = [cols[i] for i in startercols+morecols]	
#X_train, Y_train, X_test, Y_test, imp = prepdata3(data, 84000, 21000, colst)	
#X_train, Y_train, X_test, Y_test, imp = prepdata3(data, 18000, 4500, cols)	
def prepdata3(data, N_train, N_test, cols, ):

#	include metadata about which features are missing in a given row	
#	X_missing = np.array(1.0*pd.isnull(data[0:N_train+N_test][cols[0]])).reshape(N_train+N_test,1)
# 	print 'X_missing.shape ', X_missing.shape
# 	
# 	
# 	for i in range(1,len(cols)):
# 		print i
# 		X_missing = np.hstack((X_missing, np.array(1.0*pd.isnull(data[0:N_train+N_test][cols[i]])).reshape(N_train+N_test,1)))

	
	#data2 = data[:][:].fillna(-1)
 	imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #strategy='most_frequent' 'mean'
 	
 	q = 0 #50000
 	idx = range(0, N_train+N_test)
 	#idx = range(0, data.shape[0])
	random.shuffle(idx)
	idx0 = idx[0:N_train]
	idx2 = idx[N_train:N_train+N_test]
 	
 	Y = np.array(data.loss[q:q+N_train+N_test])
 	Y_train = Y[idx0]
 	Y_test = Y[idx2]
# 	Y_train = np.array(data.iloc[idx0]['loss']) 
# 	Y_test = np.array(data.iloc[idx2]['loss']) 
 	
	X = np.array(data[q:q+N_train+N_test][cols])
	
# 	X = np.hstack((X, X_missing))
 	
	X_train = X[idx0,:]
	X_test = X[idx2,:]

 	print 'imputer'
 	X_train = imp.fit_transform(X_train)
 	X_test = imp.transform(X_test)

									
	return X_train, Y_train, X_test, Y_test, imp
	
	
	

#scaler, glmf, clf, test_score = glm_clf(X_train, Y_train, X_test, Y_test, 0.66)
#classify default and non-default (glm) and use regression to predict loss values (clf)
def glm_clf(X_train, Y_train, X_test, Y_test, cutoff):	

	scaler = preprocessing.StandardScaler()
	#scaler = preprocessing.MinMaxScaler(feature_range=[0.0,100000.0])

	X_train = scaler.fit_transform(X_train).astype(np.float)
	X_test = scaler.transform(X_test).astype(np.float)

	#clf = linear_model.SGDRegressor() #loss="epsilon_insensitive"
	#clf = svm.SVR(kernel='sigmoid', gamma=0.1)
	#clf = linear_model.LogisticRegression(penalty='l1',  C=0.1) #class_weight='auto', # l1, C=1e8 takes a long time but works well
	#clf = linear_model.Ridge(alpha=0.1)	
	#clf = tree.DecisionTreeRegressor()
	#clf = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=8, random_state=0, loss='ls') #really slow
	#clf = linear_model.Perceptron(penalty='l1')
	clf = RandomForestRegressor(n_estimators=100, min_samples_leaf=12, max_features=X_test.shape[1]) #  'sqrt'
	#clf = svm.LinearSVC(penalty='l1', dual=False)

	#pos = list(np.nonzero(Y_train)[0])
	#pos = [i for i in range(len(Y_train)) if Y_train[i] != 0]


	print 'starting glm'
	
	glm = sm.GLM(1.0*(Y_train > 0), X_train[:,0::] , sm.families.Binomial())
	glmf = glm.fit()
	
	p_train = glmf.predict(X_train[:,0::])
	p_test = glmf.predict(X_test[:,0::])
 	predict_train = p_train > cutoff
 	predict_test = p_test > cutoff
#	predict_train = results.predict(new_train) > 0.5
#	predict_test = results.predict(new_test) > 0.5
	
#	auc = roc_auc_score(Y_test > 0,results.predict(new_test))
	auc = roc_auc_score(Y_test > 0,glmf.predict(X_test[:,0::]))

	print 'train: ', np.mean(predict_train == (Y_train >0))
	print 'test: ', np.mean(predict_test == (Y_test > 0))	
	print 'train score: ', np.mean(np.abs(predict_train - Y_train))
	print 'test score: ', np.mean(np.abs(predict_test - Y_test))
	print 'test auc ', auc
	print 'starting regression'
	
	pos = predict_train == 1


	clf.fit(np.hstack((X_train[pos,:], p_train[pos].reshape(-1,1))), Y_train[pos] )	
	predict_train2 = 1.0*clf.predict(np.hstack((X_train[:,:], p_train[:].reshape(-1,1) )) )

# 	clf.fit(X_train[pos,:], Y_train[pos] )	
# 	predict_train2 = 1.0*clf.predict(X_train[:,:] )

	print 'clf train score ', 1.0*np.mean(np.abs(predict_train*predict_train2 - Y_train ))  

	predict_test2 = 1.0*clf.predict(np.hstack((X_test[:,:], p_test[:].reshape(-1,1) )))
	#predict_test2 = 1.0*clf.predict(X_test[:,:] )
	test_score = 1.0*np.mean(np.abs(predict_test*predict_test2 - Y_test )) 
	print 'clf test score ', test_score 
	
	
# 	model = sm.regression.linear_model.OLS(Y_train[pos], X_train[pos,:])
# 	results = model.fit()
# 	predict_test2 = results.predict(X_test)
# 	print 'glm test score ', 1.0*np.mean(np.abs(predict_test*predict_test2 - Y_test ))  
		
		
	return scaler, glmf, clf, test_score
	



#process_test(imp, colst, scaler, glmf, clf, cutoff)
def process_test(imp, cols, scaler, glmf, clf2, cutoff):	
	
	#d = pd.read_csv('test_v2.csv',  header=0, nrows=1001)
	reader = pd.read_csv('test_v2.csv',  chunksize=10000, header=0, iterator=True)
	

	i = 0
	for chunk in reader:
# 		if i >= 2:
# 			break
		print '-----'
		#cols2 =list(chunk.columns.values)
		#print len(cols2), cols2[0], cols2[-1]
		print chunk[:][cols].shape
		print 'imputer', i
		d = convert(chunk[:][cols])
		d = imp.transform(d)	
		#d = imp.transform(chunk[:][cols])	
		#print d.shape
		d = scaler.transform(d)
# 		print 'pca', i
# 		d = pca.transform(d[:,:])
# 		print d.shape
		print 'predicting', i
		p1 = 1.0*glmf.predict(d)
		p1[np.isnan(p1)] = 0.0
		p2 = 1.0*(p1 > cutoff)
		prediction = p2*clf2.predict(np.hstack((d, p1.reshape(-1,1) )) )
		#prediction = (p1 > 0.66)*clf2.predict(d)
		#prediction = clf2.predict(d)
		
		if i != 0:
			np.savetxt(file('predict11.csv','a'), zip(chunk['id'],prediction), delimiter=",", fmt='%i, %i')
		else:
			np.savetxt("predict11.csv", zip(chunk['id'],prediction), delimiter=",", fmt='%i, %i')
		
		i = i+1