# Usage

### Training

`python train.py --config sample_config.json`

Train a **negative** auto-encoder based on data `data/yelp/sentiment.train.0`, which takes the neutral content and negative attributes as inputs,
and will output a negative comment. <br>
You can also train a positive auto-encoder by simply modifying the `"src": "data/yelp/sentiment.train.1",` in `sample_config.json`.<br>


### Testing

`python test.py --config sample_config.json`

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