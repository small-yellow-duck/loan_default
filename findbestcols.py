# Kaggle loan default prediction 
# author: small yellow duck
# this script contains functions for identifying features which are highly predictive of default/non-default


import loan_default
import numpy as np
from matplotlib import pyplot as plt

#startercols = [520,521,271,0,1,268,269]
#auc = findbestcols_glm(X_train, Y_train, X_test, Y_test, startercols)
def findbestcols_glm(X_train, Y_train, X_test, Y_test, startercols):

	auc = np.zeros((X_test.shape[1], X_test.shape[1]))
	#auc = np.zeros(X_test.shape[1])
	
	print do_glm(X_train[:,startercols+[2,3]], Y_train, X_test[:,startercols+[2,3]], Y_test)
	
	for i in range(0, X_test.shape[1]):
# 		auc[i] = do_glm(X_train[:,startercols+[i]], Y_train, X_test[:,startercols+[i]], Y_test)
# 		print i, auc[i]	
		if not i in startercols:

			for j in range(i+1, X_test.shape[1]):
				if not j in startercols:
					try:
						auc[i,j] = do_glm(X_train[:,startercols+[i,j]], Y_train, X_test[:,startercols+[i,j]], Y_test)
					except:
						auc[i,j] = 0.0
						pass
					auc[j,i] = auc[i,j]
					print i, j, auc[i,j]
		
	return auc
	


def saveauc(auc, cols):	

# 	f = open('auc_2d.txt', 'w')
# 
# 	for i in range(len(cols)):
# 
# 		f.write(cols[i] + ', ' + str(auc[i]) +'\n')
# 				
# 	f.close()
	np.savetxt('auc_2d.txt', delimiter=',')
	
	
#auc = loadauc()	
def loadauc():
# 	import csv
# 	reader = csv.reader(open('auc_2d.txt', "rb"), delimiter = ",")
# 	
# 	cols = []
# 	auc = []
# 	
# 	for row in reader:
# 		cols.append(row[0])
# 		auc.append(row[1])
		
	auc =np.loadtxt('auc_2d.txt', delimiter=',')	
		
	return auc		
	
	
	
def plot_predictive(auc):
	plt.figure(0)
	plt.clf()	
	
	x,y = np.meshgrid(range(auc.shape[0]), range(auc.shape[0]) )
	
	plt.pcolor( x,y, auc, cmap='RdBu_r', vmin=0.5, vmax=1.0)
	cbar = plt.colorbar()
	cbar.set_label('auc')
	

	plt.ylabel('feature #')
	plt.xlabel('feature #')
	#plt.axis([0,auc.shape[0],0,auc.shape[0]])
	plt.axis([200,300,450,550])
	plt.title('auc for pairs of features')
	
	plt.show()
