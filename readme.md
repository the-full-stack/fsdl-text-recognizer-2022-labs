# 🥞 Full Stack Deep Learning Fall 2022 Labs

Welcome!

As part of Full Stack Deep Learning 2022, we will incrementally develop a complete deep learning codebase to create and deploy a model that understands the content of hand-written paragraphs.

For an overview of the Text Recognizer application architecture, click the badge below to open an interactive Jupyter notebook on Google Colab:

<div align="center">
  <a href="http://fsdl.me/2022-overview"> <img src=https://colab.research.google.com/assets/colab-badge.svg width=240> </a>
</div> <br>

We will use the modern stack of [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://www.pytorchlightning.ai/).

We will use the main workhorses of DL today: CNNs and Transformers.

We will manage our experiments using what we believe to be the best tool for the job: [Weights & Biases](https://docs.wandb.ai/).

We will set up a quality assurance and continuous integration system for our codebase using [pre-commit](https://pre-commit.com/) and [GitHub Actions](https://docs.github.com/en/actions).

We will package up the prediction system as a REST API and deploy it as a [Docker](https://docs.docker.com/) container on [AWS Lambda](https://aws.amazon.com/lambda/).

We will wrap that prediction system in a frontend written in Python using [Gradio](https://gradio.app/docs).

We will set up monitoring that alerts us to potential issues in our model using [Gantry](https://gantry.io/).

# Click the badges below to access individual lab notebooks on Colab

| Notebook    | Link                                                                                                                                                                              |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Lab 00: Overview | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://fsdl.me/lab00-colab) |
| Lab 01: Deep Neural Networks in PyTorch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://fsdl.me/lab01-colab) |
| Lab 02a: PyTorch Lightning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://fsdl.me/lab02a-colab) |
| Lab 02b: Training a CNN on Synthetic Handwriting Data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://fsdl.me/lab02b-colab) |
| Lab 03: Transformers and Paragraphs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://fsdl.me/lab03-colab) |
