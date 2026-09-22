from sklearn.metrics import roc_curve,accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score,plot_confusion_matrix
from sklearn.metrics import auc
# example of k-fold cross-validation with an imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score,plot_confusion_matrix
from sklearn import metrics

from scipy import interp
# generate 2 class dataset
#y= np.argmax(y, axis=1)
#c= np.argmax(y, axis=1)
#print(y)
#X, y =make_multilabel_classification(n_samples=100, n_features=14, n_classes=14, n_labels=14)
#X, y = make_classification(n_samples=844,n_lables=14, random_state=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
# enumerate the splits and summarize the distributions
#	aucs = [] 
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
  
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
i = 1
for train_ix, test_ix in kfold.split(X):
	# select rows
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	  
	model.fit(train_X,train_y,verbose=2, epochs=100)  

	ypred = model.predict(test_X)
	fpr, tpr, _ = roc_curve(test_y[:, i], ypred[:, i])
	roc_auc = auc(fpr, tpr)
#	fpr, tpr, t = roc_curve(test_y.ravel(), ypred.ravel())
	tprs.append(interp(mean_fpr, fpr, tpr))
#	roc_auc = auc(fpr, tpr)
	aucs.append(roc_auc)
	plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
	i= i+1
	#round probabilities to class labels
	ypred = ypred.round()    
	#acc = accuracy_score(test_y, ypred)
	#print('acc is')
	#print('>%.3f' % acc)    
	print('true',test_y)
	print("pred",ypred)
	test_ycm = np.argmax(test_y, axis=1)
	ypredcm = np.argmax(ypred, axis=1)    
	acc = accuracy_score(test_ycm, ypredcm)
	cm=confusion_matrix(test_ycm, ypredcm)
	print(metrics.classification_report(test_ycm, ypredcm, digits=3))
	print(cm)
	print('acc is ',acc)
#	cm_plot_labels = ['COVID19','Pneumonia','Tuberculosis','SARS','Pneumocystis','Streptococcus','Legionella','Lipoid','Nocardia','Klebsiella','Mycoplasma Bacterial Pneumonia','todo','Cryptogenic organizing pneumonia','No finding']
#	plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
		# store result

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.text(0.32,0.7,'More accurate area',fontsize = 12)
plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()
