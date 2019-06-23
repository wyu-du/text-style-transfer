# Alter the style: A neural text style transfer model
In this project we want to build a system that can transfer a language style into another style while preserving the main content (e.g. given a factual event, we can rewrite it like a romantic story). 

**Our goal**: 

1. Transer the style (e.g. positive -> negative, factual -> romantic).

2. Preserve the content.


This task is non-trivial, because normally we do not have sufficient data that have aligned sentences with the same content but expressed in different styles. Therefore, we have to learn a model which can disentangle style attributes and content given only unaligned sentences.

**Model Architecture**: 
<br>
<img src="https://github.com/ddddwy/text-style-transfer/raw/master/data/model.png">
<br>

# Usage

### Training

```
python train.py --config sample_config.json
```

Train a **negative** auto-encoder based on data `data/yelp/sentiment.train.0`, which takes the neutral content and negative attributes as inputs,
and will output a negative comment. <br>
You can also train a positive auto-encoder by simply modifying the `"src": "data/yelp/sentiment.train.1",` in `sample_config.json`.<br>


### Testing

```
python test.py --config sample_config.json --bleu
```

Generate a negative comment based on positive comment dataset `data/yelp/sentiment.truth.1`, the procedure is as follows:

* Take a positive comment, and **extract its neutral content** as the one of the inputs;

* Retrieve a negative comment based on the neutral content, and **extract its negative attributes** as one of the inputs;

* Combine the neutral content and negative attributes together, and feed them into the well-trained model;

* Output the **generated negative comment**.


The script will generate the following files:

* `sample_run/inputs.epoch` : the input neutral contents

* `sample_run/auxs.epoch` : the input negative attributes

* `sample_run/preds.epoch` : the generated negative comments

* `sample_run/golds.epoch` : the ground truth negative comments

And you can also see the precision, recall, edit_distance and rouge score on the logging info.

# Reference

* Li, Juncen, et al. "Delete, retrieve, generate: A simple approach to sentiment and style transfer." arXiv preprint arXiv:1804.06437 (2018).

* https://github.com/lijuncen/Sentiment-and-Style-Transfer

* https://github.com/rpryzant/delete_retrieve_generate
