# pyx_COV19
Using deep transfer learning technique to detect COVID-19 disease on chest X-Rays.
\
A <em>pre-trained</em> <strong>VGG16 Convolutional Neural Network</strong> was built to particularly distinguish COVID-19 cases from Normal and Viral Pneumonia cases.


The initial data consist of 2905 images:
  - <strong>219 images</strong> of <strong>COVID-19</strong> X-rays
  - <strong>1341 images</strong> of <strong>normal</strong> X-rays
  - <strong>1345 images</strong> of <strong>viral pneumonia</strong> X-rays
  
Note: On <strong><em>December the 12th of 2020</em></strong>, the number of images in the COVID-19 category from <em>Kaggle</em> increased to 1143 and the images were resized to <strong>256x256 pixels</strong>.


Results: The <em>F1-score</em> achieved <strong>100%</strong> for the <strong>COVID-19</strong> class, <strong>96%</strong> for the <strong>Normal</strong> class and <strong>97%</strong> for the <strong>Viral Pneumonia</strong> class.
\
The <em>accuracy</em> achieved <strong>97%</strong>.


The data is availaible on: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
