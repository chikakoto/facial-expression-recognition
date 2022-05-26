# Feature learning in Facial Expression Recognition

In this project we would like to explore challenges of feature learning in facial expression recognition, and build a system capable of recognizing a type of human emotion expressed in a photo of a human face.

## Data file
Original data is from https://www.kaggle.com/datasets/msambare/fer2013.

If you have CCNY Google account, you can access the dataset from https://drive.google.com/file/d/1SWSs4FytkIPkj_v-EQaA_YKYqyrh5G6F/view?usp=sharing. 

Also, all the feature transfered data are at https://drive.google.com/file/d/1gU6ltsuIZ1UJRlE-oniPN7FRXeGotWl8/view?usp=sharing.

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── webcam         <- Webcam live facial expresssion detection script
    │       └── fer.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
