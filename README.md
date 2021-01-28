# Fake News Detector Powered By Machine Learning

A complete example of building an end-to-end machine learning project from initial idea to deployment. 

![](assets/shorter_live_run.gif)

This repo accompanies the blog post series describing how to build a fake news detection application. The posts included here:

- [Initial Setup and Tooling](https://www.mihaileric.com/posts/setting-up-a-machine-learning-project/): Describes project ideation, setting up your repository, and initial project tooling. 

- [Exploratory Data Analysis](https://www.mihaileric.com/posts/performing-exploratory-data-analysis/): Describes how to acquire a dataset and perform exploratory data analysis with tools like [Pandas](https://pandas.pydata.org/) in order to better understand the problem.

- [Building a V1 Model Training/Testing Pipeline](https://www.mihaileric.com/posts/machine-learning-project-model-v1/): Describes how to get a functional training/evaluation pipeline for the first ML model (a random-forest classifier), including how to properly test various parts of your pipeline.

- [Error Analysis and Model V2](https://www.mihaileric.com/posts/machine-learning-project-error-analysis-model-v2/): Describes how to interpret what your first model has learned through feature analysis (via techniques like [Shapley values](https://christophm.github.io/interpretable-ml-book/shapley.html)) and error analysis. Also works toward a second model powered by [Roberta](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/). 

- [Model Deployment and Continuous Integration](https://www.mihaileric.com/posts/machine-learning-project-model-deployment/): Describes how to deploy your model using [FastAPI](https://fastapi.tiangolo.com/) and [Docker](https://www.docker.com/) and build an accompanying Chrome extension. Also illustrates key components of a continuous integration system for collaborating on the application with other team members in a scalable and reproducible fashion.

## Features

* **Random forest classifier** powered by [Scikit-learn](https://scikit-learn.org/stable/).
* **RoBERTa** model powered by [HuggingFace Transformers](https://huggingface.co/transformers/) and [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).
* **Data versioning** and configurable train/test pipelines using [DVC](https://github.com/iterative/dvc).
* **Exploratory data analysis** using [Pandas](https://pandas.pydata.org/).
* **Experiment tracking** and **logging** via [MLFlow](https://mlflow.org/).
* **Continuous integration** with [Github actions](https://github.com/features/actions).
* **Functionality tests** powered by [PyTest](https://docs.pytest.org/en/stable/) and [Great Expectations](https://greatexpectations.io/).
* **Error** and **model feature analysis** via [SHAP](https://github.com/slundberg/shap).
* **Production-ready server** via [FastAPI](https://fastapi.tiangolo.com/) and [Gunicorn](https://gunicorn.org/).
* **Chrome extension** for interacting with a model in the [browser](https://chrome.google.com/webstore/category/extensions?hl=en).

## How to Use It

Go to the root directory of the repo and run:
```
pip install -r requirements.txt
```

Download the data from [this link](https://github.com/Tariq60/LIAR-PLUS/tree/master/dataset/tsv) into `data/raw`.

You're ready to go!

### Train

To train the [random forest baseline](https://www.mihaileric.com/posts/machine-learning-project-model-v1/), run the following from the root directory:
```
dvc repro train-random-forest
```

Your output should look something like the following:
```
INFO - 2021-01-21 21:26:49,779 - features.py - Creating featurizer from scratch...
INFO - 2021-01-21 21:26:49,781 - tree_based.py - Initializing model from scratch...
INFO - 2021-01-21 21:26:49,781 - train.py - Training model...
INFO - 2021-01-21 21:26:50,163 - features.py - Saving featurizer to disk...
INFO - 2021-01-21 21:26:50,169 - tree_based.py - Featurizing data from scratch...
INFO - 2021-01-21 21:26:59,360 - tree_based.py - Saving model to disk...
INFO - 2021-01-21 21:26:59,459 - train.py - Evaluating model...
INFO - 2021-01-21 21:26:59,584 - train.py - Val metrics: {'val f1': 0.7587628865979381, 'val accuracy': 0.7266355140186916, 'val auc': 0.8156070164865074, 'val true negative': 381, 'val false negative': 116, 'val false positive': 235, 'val true positive': 552}
```

### Deploy

Once you have successfully trained a model using the step above, you should have a model checkpoint saved in `model_checkpoints/random_forest`.

Now build your deployment Docker image:
```
docker build . -f deploy/Dockerfile.serve -t fake-news-deploy
```

Once your image is built, you can run the model locally via a REST API with:
```
docker run -p 8000:80 -e MODEL_DIR="/home/fake-news/random_forest" -e MODULE_NAME="fake_news.server.main" fake-news-deploy
```

From here you can interact with the API using [Postman](https://www.postman.com/) or through a simple cURL request:
```
curl -X POST http://127.0.0.1:8000/api/predict-fakeness -d '{"text": "some example string"}'
```
