# Overview

<b>tensorflow-aita-classification</b> classifies text as respectful or disrespectful baesd on Reddit submissions from [r/AmItheAsshole](https://www.reddit.com/r/AmItheAsshole/)

# Quickstart

1. Unzip the contents of the model directory such that the model directory contains assets, variables, keras_metadata.pb, and saved_model.pb.
2. Back in root, run:
<code>python -m model</code>
3. Once the model loads (a few seconds), provide text to analyze.

# Examples

Sometimes the model works well and, other times, it doesn't.

| Text                                                                         | Disrespectful Probability |
|------------------------------------------------------------------------------|---------------------------|
| I ate ice cream.                                                             | 23.2%                     |
| My son fell asleep in the car and, without his knowing, I ate his ice cream. | 88.9%                     |
| I surprised my family with presents.                                         | 98.1%                     |
| I sold my family's Christmas presents because they didn't like my meatloaf.  | 91.5%                     |

# Table of Contents

| Path               | Desc                                                                         |
|--------------------|------------------------------------------------------------------------------|
| checkpoints        | contains zip of checkpoints for verification                                 |
| model              | contains zip of model assets                                                 |
| model.py           | code to create and train the model as well as test the model with user input |
| README.md          | this README                                                                  |
| requirements.txt   | Python dependencies                                                          |

# Environment

* Microsoft Windows 10 Home
* Python 3.9.7
	* TensorFlow 2.7.0

# Installation

Just install the required Python dependencies:<br>
<code>python -m pip install -r requirements.txt</code>

# Data

* The data includes submissions to [r/AmItheAsshole](https://www.reddit.com/r/AmItheAsshole/) between 2008-01-05 (when the subreddit was first created) and 2022-01-07.
* After excluding submissions with a score less than 3, the remaining 152,545 submissions included 121,290 (79.5%) respectful submissions and 31,255 (20.4%) disrespectful submissions.
* The data requires about 268 MB of storage and is, therefore, not included in this repo. Refer to the checkpoints and model subdirectories for more info on the trained model.

# Resources
* TensorFlow Text Classification<br>
[https://www.tensorflow.org/tutorials/keras/text_classification](https://www.tensorflow.org/tutorials/keras/text_classification)

# Roadmap

* create Confusion Matrix
* create top 10 lists of respectful and disrespectful words