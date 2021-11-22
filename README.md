## _Purpose_
______
The purpose of this demo notebook is provide sample codes to create a demo that illustrate how computer vision model can be trained via transfer learning with the image data source from [Kaggle].
______
## _Background_
______
How we can know and act quickly to people with suspected COVID-19, so they can isolate themselve and alert close contact? Currently, due to Covid-19 situation, healthcare workers feel the strain of manpower crunch amid rise in [Covid-19 cases in Singapore].

Currently, a formal diagnosis of COVID-19 requires a laboratory test (RT-PCR) of nose and throat samples. RT-PCR requires specialist equipment and takes at least 24 hours to produce a result. COVID-19 is a respiratory disease. healthcare workers may use this model to diagnose people who have COVID-19 symptoms, while awaiting RT-PCR results. It very hard to diagnose huge amount of X-ray scans in short time, and take decisions deterministically at the right momemt. This model can reduce theressure on healthcare workers as COVID-19 cases spike.
______
## _What is being used a.k.a Tech_
______
**Amazon SageMaker notebook** : An Amazon SageMaker notebook instance is a machine learning (ML) compute instance running the Jupyter Notebook App. SageMaker manages creating the instance and related resources. Use Jupyter notebooks in your notebook instance to prepare and process data, write code to train models, deploy models to SageMaker hosting, and test or validate your models.

**Amazon SageMaker local mode** : Amazon SageMaker supports local mode, which allows you to create model and deploy them to your local environment. This is a great way to test your deep learning scripts before running them in SageMaker’s managed training or hosting environments. Local Mode is supported for frameworks images (TensorFlow, MXNet, Chainer, PyTorch, and Scikit-Learn) and images you supply yourself. With this, no installation is required when you use the SageMaker Notebook to run this demo.

**Pytorch** : PyTorch is an optimized tensor library for deep learning using GPUs and CPUs. It can be used for applications such as computer vision and natural language processing. This is a Library for Phyton programs that facilitates building deep learning projects. It emphasizes flexibility and allows deep learning models to be express in **idiomatic Phyton**.

**Computer Vision** : Computer vision allows machines to identify people, places, and things in images with accuracy at or above human levels with much greater speed and efficiency. Often built with deep learning models, it automates extraction, analysis, classification and understanding of useful information from a single image or a sequence of images. Medical imaging can greatly benefit from recent advances in image classification and object detection. Several studies have demonstrated promising results in complex medical diagnostics tasks spanning dermatology, radiology, or pathology. Deep-learning systems could aid physicians by offering second opinions and flagging concerning areas in images.

**Convolutional Neural Networks (CNN)** : A convolutional neural network (CNN, or ConvNet) is another class of deep neural networks. CNNs are most commonly employed in computer vision. Given a series of images or videos from the real world, with the utilization of CNN, the ML Model learns to automatically extract the features of these inputs to complete a specific task, e.g., image classification, face authentication, and image semantic segmentation. Those convolutional neural networks (CNN) have demonstrated strong performance in transfer learning, in which a CNN is initially trained on a massive dataset (e.g., ImageNet) that is unrelated to the task of interest and further fine-tuned on a much smaller dataset related to the task of interest (e.g., medical images).
______
## _Reference_
______

- https://github.com/aws-samples/amazon-sagemaker-local-mode
- https://pytorch.org/docs/stable/index.html
- https://aws.amazon.com/computer-vision/
- https://viso.ai/deep-learning/deep-neural-network-three-popular-types/
- https://www.kaggle.com/portgasray/covid-19-detection-with-x-ray-covid19-pytorch



   [Kaggle]: <https://www.kaggle.com/pranavraikokte/covid19-image-dataset>
   [Covid-19 cases in Singapore]: <https://www.straitstimes.com/singapore/health/spore-healthcare-workers-feel-the-strain-of-manpower-crunch-amid-rise-in-covid-19>
 