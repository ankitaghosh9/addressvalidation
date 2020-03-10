Address Validation Model 

Dataset: 
attribute=address,
label= invalid(1) or valid(0) address

train set:(1317231, 2) , 
test set:(520000, 2)
 
Models used:
1. Support Vector Machine (SVM)
2. Multinomial Naive Bayes (MNB)

Results:
~~~~~~~~~~SVM RESULTS~~~~~~~~~~
Accuracy Score using SVM: 97.6312
F Score using SVM:  85.0294
Confusion matrix using SVM:
[[1247280     943]
 [  30260   38748]]
 
~~~~~~~~~~MNB RESULTS~~~~~~~~~~
Accuracy Score using MNB: 96.0494
F Score using MNB: 68.7774
Confusion matrix using MNB:
[[1248135      88]
 [  51951   17057]]
