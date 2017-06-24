import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier

print('-'*90)
labeled_images = pd.read_csv("train.csv")
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images, train_labels, test_labels = train_test_split(images,labels,train_size=0.8,random_state=4)
#print(train_images>0)
#for a in train_images:
	
#	print('-'*90)
#	for b in range(0,784):
#		if a[['pixel'+'b']]>0:
#			a[['pixel'+'b']]=1
		
#for a in test_images:
#	for b in range(0,784):
#		if a[['pixel'+'b']]>0:
#			a[['pixel'+'b']]=1


#test_images.loc[test_images>0]=1
#train_images.loc[train_images>0]=1
print(test_images.describe())
#for i in range(0,3):
#i = 1
#    img = train_images.iloc[i].as_matrix()
 #   img = img.reshape((28,28))
 #   plt.imshow(img,cmap = "binary")
 #   plt.title(train_images.iloc[i,0])
 #   plt.show()
nn = MLPClassifier(activation="logistic",solver="sgd",hidden_layer_sizes=(20,20),random_state=1)
nn.fit(train_images,train_labels.values.ravel())
print(nn.score(test_images,test_labels))



#clf = svm.SVC()
#clf.fit(train_images,train_labels.values.ravel())
#clf.score(test_images,test_labels)


test_data=pd.read_csv('test.csv')
#test_data[test_data>0]=1
results=nn.predict(test_data[0:])
out = pd.DataFrame(results)

out.index+=1
out.index.name='ImageId'
out.columns=['Label']
out.to_csv('results.csv', header=True)
print('-'*90)