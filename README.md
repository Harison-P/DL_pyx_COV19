# pyx_COV19
<em>A journey with artificial intelligence in service of medical diagnosis</em>

This project was performed during a Bootcamp Data Science training with DataScientest, by Kevin Pame and Valerie Ducret. It is above all a didactic project that finally occurs to be really efficient, with a recall of 100% for COVID-19 labeled X-rays. 
By deploying a web application using Streamlit, we hope that others would be able to learn about Computer Vision, understand how a neural network works, or use some of this work for their own data science project.

The data is availaible on: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

# Explanation of DL_pyx_COV19 repository

## Folders

The <em>dataset</em> folder contains the data used to train/test our model, with one subfolder for each category (COVID-19, Normal, Viral Pneumonia). It was not possible to upload all X-rays, but the original dataset is indicated in the readme.md file included in the folder.
\
The <em>modules</em> folder contains all the necessary dependencies to run specific functions during the navigation in the web app:

  - <strong>features_map.py</strong> to generate a features map 
  - <strong>img_classification.py</strong> to get the predicted label of an uploaded X-ray
  - <strong>grad_cam.py</strong> to generate a GRAD-CAM visualization

One folder contains some X-rays to test the model's prediction. The Covid X-rays was extracted from a scientific paper published by Elsevier
\
The "static" folder contains images displayed in the web app such as the VGG16 architecture or the confusion matrices.

## Main directory

Our final CNN model used in this web app is entitled "fine_tuned_vgg16_second_model" and is attached to the main directory.
\
Our script used to deploy the Streamlit app is entitled <strong>xray_app.py</strong>

# Introduction to the project

Coronaviruses are a diverse group of viruses infecting many different animals, and they can cause mild to severe respiratory infections in humans (Hu et al., 2020). In late 2019, a novel coronavirus designated as SARS-Cov-2 emerged in the city of Wuhan (China) and caused an outbreak of unusual viral pneumonia, called COVID-19, that rapidly developed as an unprecedented world sanitary crisis. One year after, about 74 million people all other the world has contracted COVID-19, and about 1.7 million people have died from it. This highly contagious virus, by binding to epithelial cells in the respiratory tract, starts replicating and migrating down to the airways and enters alveolar epithelial cells in the lungs. Fast replication of SARS-CoV-2 in the lungs may trigger a strong immune response (Hu et al., 2020). Thus, cytokine storm syndrome causes acute respiratory distress syndrome and respiratory failure, which is considered the main cause of death in patients with COVID-19 (Huang et al., 2020; Mehta et al., 2020). Therefore, this outbreak has led to drastic human and economic consequences for which we still do not gauge final magnitude. Despite public health responses trying to decrease contamination rate, the rapidly evolving situation conducted to a saturation of hospitalization demands and an increase in mortality rate.


  One efficient measure to contain the disease and delay the spread is active testing for COVID-19. By identifying contaminated people, public organization can take measures to isolate and limit contacts with the infected ones. Indeed, without testing, it is impossible to discriminate COVID-19 from other virus like influenza responsible for the ?flu?, because symptoms are very similar depending on the severity of COVID-19. The gold-standard tests that detect viruses are RT-PCR (Real-Time Polymerase Chain Reaction). The high-sensitivity of PCR tests are almost 100% accurate in spotting infected people, when they are administered properly. But such tests generally require trained personnel, specific reagents that are lacking in periods of high demands, and expensive machines that take time to provide results. Therefore, other methods were developed to detect COVID-19 rapidly and efficiently. For instance, antigen assays are faster and cheaper than PCR tests, but are not as sensitive and could miss infectious people. Other alternative methods incorporating analysis of chest radiographies (computed tomography CT-scans and X-rays) may assist in identifying false negative RT-PCR cases or when RT-PCR tests are unavailable. However, it is tricky and time-consuming for radiologists to discriminate COVID-19 from other viral pneumonia by eye.
  

  Today, computer vision is assisting an increasing number of doctors to better diagnose their patients, monitor the evolution of diseases, and prescribe the right treatments. It is an emerging field that takes advantage of artificial intelligence algorithms that process images and often make a faster and more accurate diagnosis than humans could do. Potential application of computer vision systems is minimizing false positives in the diagnostic process or detect the slightest presence of a condition. One difficulty for radiologists comes down to discriminating between a classic viral pneumonia and pneumonia caused by COVID-19. A study demonstrates that radiologists had high specificity (true negative rate) but moderate sensitivity (true positive rate) in differentiating COVID-19 from viral pneumonia on chest CT-scans (Bai et al., 2020). Also, analyses of chest X-rays and CT-scans may show different sensitivities in detecting COVID-19. Despite that chest X-ray abnormalities of COVID-19 mirror those of CT-scans, less dense opacities may be more difficult to detect by eye and conduct to a sensitivity of 69% compared to generally more than 90% for CT-scans (Wong et al., 2020). Therefore, the use of such methodology could highly help mitigate the burden on the healthcare system by providing accurate models that detect COVID-19 on CT-scans (Ahuja et al., 2020) and particularly on chest X-rays that show moderate sensitivity.


  The objective of this project is to build a multiclass classification model that can accurately predict COVID-19 from chest X-rays, and particularly discriminate from viral pneumonia or X-rays taking from healthy patients.
  

## Data

The initial data consist of 2905 images:

  -	219 images of COVID-19 X-rays (which represent about 7.5% of total data)
  -	1341 images of normal X-rays (which represent 46.2% of total data)
  -	1345 images of viral pneumonia X-rays (which represent 46.3% of total data)

The initial data have a size of 1024x1024 pixels. Most of the images are in grayscale but few of them show blue hues.
\
Note: On December the 12th of 2020, the number of images in the COVID-19 category from Kaggle increased to 1143 and the images were resized to 256x256 pixels.

No issues were come across upon the study of the data. Although at first, when we proceeded to some preliminary analysis, we noticed that some images were duplicated. How to verify if two images are identical? The signature of an image resides in its array. For a computer, an image is perceived as an array or matrix of values ranging from 0 to 255. By comparing all the values of a matrix element by element to another matrix, we would check whether two images are identical.
We then decided not to pursue further with the idea of dropping duplicated images as we would if we had to deal with more structured data. The first reason is that this preliminary analysis was time consuming for a result that was not significant enough, that is to say that, in reality, less than 10 duplicated images were found throughout the Normal and Viral Pneumonia categories and less than 25 duplicated images were found in the COVID-19 category.
The second reason, as one may notice, comes from the fact that the data is unbalanced in favor of the main target variable which is the COVID-19 label. To compensate for the unbalanced dataset, two ideas were opted, namely data augmentation and adjusting class weights.
Data augmentation corresponds to a data analysis technique that is used to increase the amount of data by modifying copies of already existing data. These modifications consist of applying geometric transformations such as flipping, cropping, rotation etc. As for the second idea, we added more weight to the COVID-19 category so that the model puts much more emphasis during training. Otherwise, this category would have been missed out and the opportunity of detecting COVID-19 cases would have fallen short of expectations.

The data were split in two parts with respect to the proportion of data between classes:
-	The training set amounts to 80% of the total data
-	The test set amounts to 20% of the total data
