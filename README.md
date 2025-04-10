# Fashion-Recommendation-System-Using-CNN

PROJECT INTRODUCTION
This chapter provides an overview of the project, detailing its objectives, scope, and significance within the broader context of the subject. 
2.1 PROJECT OBJECTIVES
With the rising standard of living, people's attention has increasingly shifted towards fashion, a popular aesthetic expression. Humans are naturally drawn to visually appealing items, which has fueled the rapid growth of the fashion industry. However, the abundance of garment options available on e-commerce platforms has introduced challenges for customers in identifying the right outfit.
The Fashion Recommender System project aims to address this issue by developing an innovative solution that enhances the online shopping experience. Unlike traditional systems that rely on user purchase history and preferences, this system focuses on generating recommendations based on visual similarity. Users can provide an image of a product they are interested in, and the system uses advanced neural networks to process the input and recommend visually similar items.
The project's key objectives are:
Enhancing the Shopping Experience: Simplify the process of discovering visually appealing fashion items, ensuring users find what they are looking for more intuitively.
Innovative Technology Integration: Leverage neural networks and content-based recommendation techniques to process product images from the Fashion Product Images Dataset.
Eliminating Historical Dependency: Provide recommendations based on image content, reducing reliance on user purchase history or preferences.
Accuracy and Personalization: Use a nearest-neighbor algorithm to ensure precise, personalized recommendations aligned with user interests.
Scalable Solution: Design a system capable of adapting to diverse datasets and evolving user demands in the dynamic fashion industry.

2.1.1 PROBLEM STATEMENT 
The online fashion industry presents a unique challenge: an overwhelming variety of options often leaves customers struggling to find their desired products. Traditional recommendation systems, which rely heavily on user purchase history or explicit preferences, fail to capture the nuanced requirements of users searching for similar items based on visual attributes.
Solution
The Fashion Recommender System addresses this gap by using an image-based approach to provide personalized recommendations:
Content-Based Recommendations: Utilizes convolutional neural networks (CNNs) to extract semantic features from product images, enabling the detection of visually similar items.
Efficient Similarity Detection: Employs techniques like cosine similarity and nearest-neighbor algorithms to rank products based on visual proximity to the user’s input image.
Advanced Neural Network Integration: Incorporates pre-trained models like ResNet50, fine-tuned for the fashion domain, to generate high-quality embeddings.
Interactive Interface: Allows users to upload an image and receive recommendations seamlessly, improving decision-making during online shopping.

2.2.2 SYSTEM ARCHITECTURE :
In this project, we propose a model that uses Convolutional Neural Network and the Nearest neighbour backed recommender. As shown in the figure Initially, the neural networks are trained and then an inventory is selected for generating recommendations and a database is created for the items in inventory. The nearest neighbour’s algorithm is used to find the most relevant products based on the input image and recommendations are generated.


2.2 MAIN COMPONENTS OF PROJECTS
1. Dataset and Data Handling
Dataset
The "Fashion Product Images Small dataset from Kaggle is used as the primary data source.
The dataset includes metadata like garment type, category, and images of fashion products.
Data Preprocessing:
Images are resized, normalized, and converted for compatibility with the neural network.
Metadata is cleaned and structured using Python libraries like Pandas and NumPy.
Integration with Google Drive:
Google Drive is used to store and retrieve the dataset, ensuring easy accessibility during development.
2. Image Processing Module
Image Loading and Preprocessing:
Images are loaded using OpenCV and resized to a fixed dimension (e.g., 224x224 pixels) for input into the machine learning model.
Functions like color conversion (BGR to RGB) are applied to maintain consistency.
Feature Visualization:
Tools like Matplotlib are used to display images and verify the preprocessing steps.
3. Deep Learning Model
Pre-trained Model (ResNet50):
ResNet50, a robust convolutional neural network, is employed for extracting embeddings from garment images.
The model is pre-trained on the ImageNet dataset and fine-tuned for this project to capture meaningful visual features.
Embedding Layer:
A GlobalMaxPooling2D layer is added to compress the feature map into a dense, low-dimensional embedding vector that represents each image.
Transfer Learning:
Transfer learning is used to adapt ResNet50 for the fashion dataset, ensuring high accuracy even with a smaller dataset.
4. Recommendation Engine
Similarity Computation:
The cosine similarity metric is used to measure the similarity between embeddings.
Scikit-learn’s pairwise distance functions compute the similarity between input embeddings and the dataset.
Nearest Neighbor Search:
A Nearest Neighbor algorithm retrieves the top N similar items for a given input image based on cosine similarity scores.
Top Recommendations:
The engine generates a ranked list of similar garments, displaying the top recommendations.
5. Visualization and Dimensionality Reduction
t-SNE Visualization:
The t-SNE algorithm reduces the dimensionality of embeddings, enabling a 2D or 3D visualization of relationships between garments.
This visualization aids in understanding the clustering of similar products.
Plotting Tools:
Scatterplots and bar charts created with Matplotlib and Seaborn showcase the distribution of categories and similarity relationships.
6. User Interface
Image Input:
The system accepts an image uploaded by the user as the input query.
Output Display:
Displays the input image alongside the top recommended products for easy comparison.
Google Colab Integration:
The user interacts with the system through a Colab notebook interface, which allows for easy execution and visualization of results.
7. Evaluation and Optimization
Evaluation Metrics:
Metrics such as cosine similarity, accuracy, and F1-score are used to evaluate the system's performance.
Hyperparameter Tuning:
Parameters like embedding size, similarity threshold, and number of nearest neighbors are fine-tuned for optimal results.
8. File Management and Outputs
Embedding Storage:
Image embeddings are stored in a structured format (e.g., CSV) for efficient retrieval and analysis.
Metadata Export:
Metadata and recommendation results are saved as CSV files for future use or further analysis.


2.3 TECHNOLOGY USED
Google Colab
Google Colab (short for Google Colaboratory) is a free, cloud-based platform that provides an environment for writing, running, and sharing Python code. It is particularly well-suited for projects that involve machine learning, data analysis, and deep learning due to its powerful computing capabilities and seamless integration with popular libraries.

Key Features of Google Colab
Free Access to GPU/TPU:
Provides free access to powerful hardware accelerators like GPUs and TPUs, which significantly improve the speed of training and inference for machine learning models.
Cloud-Based:
Since Google Colab operates on the cloud, it eliminates the need for local hardware resources, making it accessible from any device with internet connectivity.
Integration with Google Drive:
Allows seamless access to datasets and storage for outputs by directly integrating with Google Drive. This simplifies data management and sharing.
Pre-installed Libraries:
 Python libraries such as TensorFlow, Keras, and Scikit-learn, reducing the need for manual installation and setup.
Interactive Notebook Environment:
Features an intuitive notebook-style interface where code, outputs, and visualizations can be displayed in a single document. This facilitates debugging, visualization, and collaboration.
Collaboration:
Enables real-time sharing and collaborative editing of notebooks, making it easier for teams to work on the project simultaneously.
Usage in the Project:
The entire Fashion Recommender System was implemented and executed in Google Colab.
The platform was used for loading and preprocessing datasets, training machine learning models, visualizing results, and exporting outputs like embeddings and recommendation results.
GPU acceleration provided by Colab was instrumental in efficiently training deep learning models like ResNet50.

Python
Python is the core programming language used for this project due to its simplicity, flexibility, and extensive ecosystem of libraries designed for machine learning and data processing.
Key Features of Python
Rich Ecosystem:
Python boasts a vast collection of libraries tailored for data science and machine learning, such as TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.
Ease of Use:
Its straightforward syntax and readability make Python an ideal choice for complex projects, reducing the time and effort required to write and debug code.
Integration:
Python integrates seamlessly with cloud platforms like Google Colab, enabling the smooth execution of resource-intensive tasks.
Extensive Community Support:
Python's large user base ensures readily available resources, tutorials, and troubleshooting solutions.
Usage in the Project:
Data Processing:
Python libraries like Pandas and NumPy were used to preprocess metadata and handle large arrays efficiently.
Image Processing:
OpenCV and Matplotlib facilitated the loading, manipulation, and visualization of garment images.
Machine Learning:
TensorFlow and Keras powered the deep learning models, particularly the pre-trained ResNet50 used for feature extraction and embeddings.
Scikit-learn was employed for similarity computation and recommendation generation.
Visualization:
Matplotlib and Seaborn were used to plot data distributions and t-SNE-based visualizations of the embedding space.

Why Python was Chosen ?
Python's versatility and extensive library support made it the natural choice for implementing a project requiring data handling, image processing, and deep learning.

2.3.1 LIBRARIES
Libraries Used in the Fashion Recommender System
The implementation of the Fashion Recommender System involves a variety of Python libraries that cater to different aspects of the project, from data preprocessing and visualization to deep learning and recommendation generation. Below is a detailed analysis of the libraries used:
1. Data Processing
Pandas: Used for loading, cleaning, and manipulating data, such as reading CSV files (styles.csv) and structuring metadata. It simplifies operations like filtering rows, creating new columns (e.g., image paths), and handling large datasets.
NumPy: Facilitates numerical computations and array manipulations. It is essential for handling and transforming image data and embeddings into numerical formats required for machine learning algorithms.
OS: Enables interaction with the file system, such as accessing directories and managing dataset paths.
2. Image Processing
OpenCV (cv2): Handles loading and resizing of garment images for uniform input dimensions to the deep learning model. It provides utilities for color space conversion (e.g., BGR to RGB) and visualization of image preprocessing steps.
Matplotlib Image: Used to plot and display images during debugging and feature verification processes.
3. Visualization
Matplotlib: The core library for creating static, animated, and interactive visualizations. It is used to display image samples and plot distribution graphs, such as the frequency of garment categories.
Seaborn: Built on Matplotlib, it provides a high-level interface for drawing attractive and informative statistical graphics. It is utilized for scatterplots and clustering visualizations, such as t-SNE-based embeddings.
4. Machine Learning and Deep Learning
TensorFlow and Keras:
TensorFlow: A robust framework for building and deploying machine learning models. It is used for creating, training, and deploying the ResNet50 model and feature extraction pipeline.
Keras: A high-level API within TensorFlow that simplifies deep learning implementation. It is leveraged for importing the pre-trained ResNet50 model and adding custom layers for embedding extraction.
Scikit-learn: Used for implementing recommendation algorithms and evaluating similarity measures.
Pairwise Distances: Calculates cosine similarity between image embeddings to identify the most similar items.
t-SNE: Reduces the dimensionality of embeddings for intuitive 2D or 3D visualization of garment similarities.
5. Performance Optimization
Swifter: A library that accelerates the application of functions across Pandas DataFrames using parallel processing. It is applied to compute embeddings and similarity measures efficiently over large datasets.
6. Dataset Management
KaggleHub: Used to download the "Fashion Product Images Small" dataset directly from Kaggle for easy integration into the project.
7. Utility Libraries
Matplotlib.pyplot: The primary module for plotting graphs and visualizing results.
Time: Measures the execution time of key processes, such as t-SNE computations.

2.3.2 MODULES USED
CATEOGRY
MODULES
Data Handling and Manipulation





Pandas, NumPy, OS



Image Processing
OpenCV (cv2), Matplotlib.image
Visualization
Matplotlib.pyplot, Seaborn
Machine Learning and Deep Learning
TensorFlow, Keras, Scikit-learn, t-SNE
Performance Optimization
Swifter
Time Management and Execution
Time
Dataset Management





KaggleHub



Neural Network
Keras.applications, Keras.layers,
Keras.preprocessing.image
Recommendation and Similarity
Scikit-learn.metrics.pairwise


2.3.3 CODE EDITOR
Google Colab
Google Colab (short for Google Colaboratory) is a cloud-based code editor and notebook environment that was utilized to write, execute, and train the Fashion Recommender System. It serves as an excellent platform for machine learning and data science projects due to its robust features and user-friendly interface. Below is a detailed overview of Google Colab and its contributions to this project:
Key Features of Google Colab
Cloud-Based Environment: Google Colab eliminates the need for powerful local hardware by providing a cloud-based environment where computations run on Google's servers. This accessibility allows users to work from any device with an internet connection.
Free Access to Hardware Accelerators: The platform supports GPUs (Graphics Processing Units) and TPUs (Tensor Processing Units) at no cost, which are essential for training computationally intensive deep learning models like ResNet50.
Integration with Google Drive: Google Colab seamlessly integrates with Google Drive, facilitating easy storage and retrieval of datasets, models, and outputs. This feature allowed for efficient handling of large files, such as images and embeddings, throughout the project.
Interactive Notebook Interface: The platform combines code execution, documentation, and visualizations in a single interactive notebook. This setup facilitates real-time debugging, exploration, and presentation of results.
Pre-installed Libraries: Google Colab comes with pre-installed libraries such as TensorFlow, Keras, Pandas, and Scikit-learn, which reduces setup time and complexity for users.
Custom Installations: Google Colab supports custom library installations using !pip install, providing flexibility in adding dependencies like Swifter and KaggleHub.

Role of Google Colab in the Fashion Recommender System
Code Development: Google Colab provided an environment to write and execute Python code for all stages of the project, including data preprocessing, model training, and recommendation generation.
Model Training: The platform's GPU support significantly reduced the time required to train the ResNet50 model and generate embeddings for the dataset.
Data Management: The integration with Google Drive allowed for easy access to the "Fashion Product Images Small" dataset and facilitated the saving of processed outputs, such as embeddings and metadata.
Visualization: Google Colab was used to display images, plots, and t-SNE visualizations directly in the notebook, ensuring real-time analysis and debugging.
Experimentation: The platform facilitated iterative experimentation by allowing quick code modifications and immediate feedback through live outputs.
Advantages of Using Google Colab
There is no need for expensive hardware, as all computations are performed in the cloud.
The platform offers easy collaboration and sharing capabilities.
Users have free access to powerful GPUs and TPUs for accelerated processing.
The interactive and user-friendly interface allows for the combination of code, visuals, and explanations in a single document.
