# Import necesary modules.
import math
from numpy import inf
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


	
#Input: Numeric values v1 and v2; Error margin for equality comparisson.
#Output: True if the ratio between both numbers is within [1-margin,1]
#        else False.
def equal(v1, v2, margin):
	r = -1
	if v1 == 0 and v2 == 0:
		return True
	elif v1 == 0 and v2 < 0:
		return False
	elif v2 == 0 and v1 < 0:
		return False
	elif v2 < 0 and v1 < 0:
		if v1 > v2:
			r = v1/v2
		else:
			r = v2/v1
	else:
		if v1 < v2:
			r = v1/v2
		else:
			r = v2/v1
	margin = 1-margin
	return r >= margin

#Input: 2D points p1 and p2.
#Output: Distance between p1 and p2.
def distance(p1,p2):
	x1, y1 = p1
	x2, y2 = p2
	
	dx = float(x1-x2)
	dy = float(y1-y2)
	
	return math.sqrt(dx*dx + dy*dy)

#Input: Contour cntr; Vertexes indexes i1 and i2.
#Output: Distance between the contour vertexes with indexes i1 and i2.
def edgeLength(cntr,i1,i2):
	p1 = cntr[i1][0]
	p2 = cntr[i2][0]
	
	return distance(p1,p2)

#Input: 2D points p1 and p2.
#Output: Slope of rect between p1 and p2.
def slope(p1,p2):
	x1, y1 = p1
	x2, y2 = p2
	
	num = (float(y1)-float(y2))
	den = (float(x1)-float(x2))
	
	if den != 0:
		return num/den
	else:
		if (y1 > y2):
			return -inf
		else:
			return inf
			
#Input: Contour cntr; Vertexes indexes i1 and i2.
#Output: Slope of rect between the contour vertexes with indexes i1 and i2.
def edgeSlope(cntr,i1,i2):
	p2 = cntr[i2][0]
	
	return slope(p1,p2)

#Input: Contour cntr; Vertexes indexes i.
#Output: Countour's internal angle in vertex with index i.
def internalAngle(cntr,i):
	n = len(cntr)
	
	p1 = cntr[(i-1)%n][0]
	p2 = cntr[i%n][0]
	p3 = cntr[(i+1)%n][0]
	
	convex = convexAngle(p1,p2,p3)
	
	x1, y1 = p1
	x2, y2 = p2
	x3, y3 = p3
	
	n1 = distance(p1,p2)
	n2 = distance(p2,p3)
	
	cos = ((x1-x2)*(x3-x2)+(y1-y2)*(y3-y2))/(n1*n2)
	ang = math.acos(cos)
	
	if not convex:
		ang = (2*math.acos(-1)) - ang
	
	return ang*180/math.acos(-1)

#Input: 2D points p1, p2 and p3
#Output: True if the angle formed by vectors p2p1 and p2p3 is less than 180 degrees,
#        False if the angle is more or equal than 180 degrees
def convexAngle(p1,p2,p3):
	x1, y1 = p1
	x2, y2 = p2
	x3, y3 = p3
	
	m1 = slope(p1,p2)
	m2 = slope(p2,p3)
	
	q1 = 0
	q2 = 0
	
	dx1 = 1 if x1 <= x2 else -1
	dy1 = 1 if y1 <= y2 else -1
	
	if dx1 == 1 and dy1 == 1:
		q1 = 1
	elif dx1 == -1 and dy1 == 1:
		q1 = 2
	elif dx1 == -1 and dy1 == -1:
		q1 = 3
	else:
		q1 = 4
	
	dx2 = 1 if x2 <= x3 else -1
	dy2 = 1 if y2 <= y3 else -1
	
	if dx2 == 1 and dy2 == 1:
		q2 = 1
	elif dx2 == -1 and dy2 == 1:
		q2 = 2
	elif dx2 == -1 and dy2 == -1:
		q2 = 3
	else:
		q2 = 4
		
	if q1 == 1:
		if q2 == 1:
			return m1 > m2
		elif q2 == 2:
			return False
		elif q2 == 3:
			return m1 < m2
		elif q2 == 4:
			return True
	elif q1 == 2:
		if q2 == 1:
			return True
		elif q2 == 2:
			return m1 > m2
		elif q2 == 3:
			return False
		elif q2 == 4:
			return m1 < m2
	elif q1 == 3:
		if q2 == 1:
			return m1 < m2
		elif q2 == 2:
			return True
		elif q2 == 3:
			return m1 > m2
		elif q2 == 4:
			return False
	elif q1 == 4:
		if q2 == 1:
			return False
		elif q2 == 2:
			return m1 < m2
		elif q2 == 3:
			return True
		elif q2 == 4:
			return m1 > m2
	return False

#Input: Contour cntr.
#Output: True if figure formed by the contour is convex,
#        False if figure is concave.
def isConvex(cntr):
	n=len(cntr)
	for i in range(n):
		p1 = cntr[(i-1)%n][0]
		p2 = cntr[i%n][0]
		p3 = cntr[(i+1)%n][0]
		if not convexAngle(p1,p2,p3):
			return False
	return True

#Input: Confusion matrix cm; Classes for plotting in alfabetical order; Boolean to select normalization;
#       Title of the plot; Color map cmap.
#Output: Shows plot of confusion matrix.
def plot_confusion_matrix(cm, classes=['0','1','2','3','4','5','6','7','8','9'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
