# Breast-Cancer-Detection-Using-CNN


Convolutional Neural Network (CNN):
     A convolutional neural network (CNN) is a deep-learning framework that takes an image, assigns weights and biases to distinct features in the image, and distinguishes one image from the other. CNN architecture is built on three main design concepts: local receptive fields, weight sharing, and sub-sampling. The CNN was initially developed to identify two-dimensional image patterns. A CNN is made up of three layers: (1) convolution layers, (2) max-pooling layers, and (3) an output layer.

![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/efac7174-7ff0-4edc-9ffb-77ebb7add166)

 
     A detailed overview of the convolutional neural network (CNN) is needed since it is a central tool in BrC classification. CNNs are utilized more often to develop a reliable BrC classification model in previous studies. CNNs are utilized with different imaging modalities as they work well with images. However, a large number of images are needed to train a CNN. It is difficult to achieve good performance with a limited number of images. Moreover, it is difficult to obtain adequate training data because obtaining labeled datasets in medical imaging is costly. However, CNN has many advantages. Opposed to other classification methods, a ConvNet requires much less pre-processing. Feature extraction and classification are entirely combined into a single CNN architecture. Lastly, it is resistant to picture noise and local geometric distortions. Therefore, studies used CNNs to extract useful features from medical images and to perform BrC classification with them. De-novo CNNs (CNNs trained from scratch) and TL-based CNNs (pre-trained CNNs) are mainly used in BrC classification.

SYSTEM MODULES:

A.	Pattern Recognition Network Artificial neural networks are useful for pattern matching applications. Pattern matching consists of the ability to identify the class of input signals or patterns. (fig..1)

![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/535dce05-1df8-40f3-8f72-fb0d367def79)

                   
B.	Training, Validation, Testing 
• The data used to build the final model usually comes from multiple datasets. In particular. Three data sets are commonly used in different stages of the creation of the model.
 • The model is initially fit on a training dataset that is a set of examples used to fit the parameters of the model.
 • In practice, the training dataset often consist of pairs of an input vector (or scalar) and the corresponding output vector (or scalar), which is commonly denoted as the target (or label). The current model is run with the training dataset and produces a result, which is then compared with the target, for each input vector in the training dataset. Based on the result of the comparison and the specific learning algorithm being used, the parameters of the model are adjusted. The model fitting can include both variable selection and parameter estimation.



 • Successively, the fitted model is used to predict the responses for the observations in a second dataset called the validation dataset. The validation dataset provides an unbiased evaluation of a model fit on the training dataset while tuning the model's hyperparameters (e.g. the number of hidden units in a neural network). Validation datasets can be used for regularization by early stopping: stop training when the error on the validation dataset increases, as this is a sign of overfitting to the training dataset. This simple procedure is complicated in practice by the fact that the validation dataset's error may fluctuate during training, producing multiple local minima. This complication has led to the creation of many ad-hoc rules for deciding when overfitting has truly begun.
 • Finally, the test dataset is a dataset used to provide an unbiased evaluation of a final model fit on the training dataset. When the data in the test dataset has never been used in training (for example in cross-validation), the test dataset is also called a holdout dataset.
 C. Train the network Plot 
Trainstate (tr) plots the training state from a training record tr returned by train. (fig.2) [net, tr] = train (net, inputs, targets)

![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/35c91fbe-cb3b-4345-b40e-8d8b4f1eda07)

 
D. Test the network 
1) Performance Plot perform (TR) plots error vs. epoch for the training, validation, and test performances of the training record TR returned by the function train. (fig. 3) outputs = net (inputs); errors = gsubtract(target, outputs); performance = perform (net, targets, outputs);

![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/0fb2c74f-fb9a-4fbb-a3e0-4f08eb8db426)

 
2) Error Histogram
 • ploterrhist(e) plots a histogram of error values e. 
• ploterrhist(e1,'name1',e2,'name2',...) takes any number of errors and names and plots each pair.
 • ploterrhist(...,'bins',bins) takes an optional property name/value pair which defines the number of bins to use in the histogram plot. The default is 20. (fig.4)
• Plot confusion (targets, outputs) plots a confusion matrix for the true labels targets and predicted labels outputs. Specify the labels as categorical vectors, or in one-of-N (one-hot) form.

![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/63d4a70e-c814-4ce1-8419-3b6cdeb0b694)

                                 
 • On the confusion matrix plot, the rows correspond to the predicted class (Output Class) and the columns correspond to the true class (Target Class). The diagonal cells correspond to observations that are correctly classified. The off-diagonal cells correspond to incorrectly classified observations. Both the number of observations and the percentage of the total number of observations are shown in each cell. 
• The column on the far right of the plot shows the percentages of all the examples predicted to belong to each class that are correctly and incorrectly classified. These metrics are often called the precision (or positive predictive value) and false discovery rate, respectively. The row at the bottom of the plot shows the percentages of all the examples belonging to each class that are correctly and incorrectly classified. These metrics are often called the recall (or true positive rate) and false negative rate, respectively. The cell in the bottom right of the plot shows the overall accuracy.
 • plot confusion (targets, output, name) plots a confusion matrix and adds name to the beginning of the plot title. 
• plot confusion (targets1, outputs1, name1, target s2, outputs2, name2, . ..,targetsn, outputsn, name n) plots multiple confusion matrices in one figure and adds the name arguments to the beginnings of the titles of the corresponding plots. 



A sequence diagram is a kind of interaction diagram that shows how processes operate with one another and in what order. It is a construct of a Message Sequence Chart. A sequence diagram shows object interactions arranged in time sequence Sequence diagrams are typically associated with use case realizations in the Logical View of the system under development. Sequence diagrams are sometimes called event diagrams, event scenarios, and timing diagrams.


![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/d4acda6f-9f03-4fa1-a24c-fcb9d56713f1)



A deployment diagram in the Unified Modeling Language models the physical deployment of artifacts on nodes. To describe a web site, for example, a deployment diagram would show what hardware components ("nodes") exist (e.g., a web server, an application server, and a database server), what software components ("artifacts") run on each node (e.g., web application, database), and how the different pieces are connected (e.g. JDBC, REST, RMI).

![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/dd85f701-97d3-406c-bbcd-1c7cd4d65258)

The proposed system will use machine learning algorithms to classify data from the Wisconsin Breast Cancer dataset, which includes 11 integer values and a class label indicating whether a person is normal (0) or infected with disease (1). Support Vector Machine (SVM) and Artificial Neural Networks (ANN) will be employed for training. SVM works by creating a hyperplane that optimally separates the classes in the dataset, ideal for both linear and non-linear problems. ANN, on the other hand, is a model that mimics the structure and function of biological neural networks, learning from data samples to predict outcomes effectively. Additionally, Convolutional Neural Networks (CNNs) will be used for image-based detection of breast cancer, utilizing layers like convolutional, pooling, and fully connected layers to process and classify image data by recognizing patterns and features. This integrated approach aims to leverage the strengths of each algorithm to improve accuracy and efficiency in disease prediction and classification.


![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/9bd7a25f-1c02-414f-9ae6-62a4a6cedfc7)



Genrate train and test model and we will findtotal test sets

![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/05de991e-9b85-4492-9a75-b72220844792)

click on ‘Preprocess Dataset’ button to read all reviews from dataset and then apply Preprocess steps

![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/3209c8da-d66d-4ae8-9492-3f5552d6e9e3)


click on “ Run CNN Algorithm” 

![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/9be9f2e7-15fe-48a3-b6a3-4dcdfdbf47a4)


click on “ Run ANN algorithm”

![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/c87cd7f8-6486-4f56-8d4c-3bfc57e6144e)


click on “Upload test data and predict disease”

![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/aa9fc4bb-221c-4df4-8144-d535b04272dc)


click on ‘Accuracy Graph’ button to get above graph

![image](https://github.com/pavanmuthyam/Breast-Cancer-Detection-Using-CNN/assets/87929903/ea46d2a0-221b-40d7-86f2-43af2555d20a)












