"""
Virtual environment.
activate py27 (conda)
C:\users\hp user\news_app\scripts   --> type activate in shell.
"""

#import pandas as pd
import numpy as np
import random
import os
import csv
#import requests
#import urllib
import json
import webapp2
#from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn.linear_model import LogisticRegression, Ridge
#from sklearn.externals import joblib
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import f1_score, confusion_matrix
#from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.embed import components, json_item
from bokeh.resources import INLINE
import joblib
from bokeh.models.glyphs import Line, Text
from collections import Counter
import math

def only_ascii(s):
    return "".join([c for c in list(s) if ord(c) <= 128])


def train_model():
    """ train machine learning model to take the text of a post as input, and,
        predict its bias label from 1 (very left) --> 3 (neutral) --> 5 (very right)

        outputs:
            model.pkl
            vectorizer.pkl
    """

    """ 
    Load news corpus and represent in a format suitable for machine learning.
    - tf-idf embed text
    - encode categorical labels numerically.
    """

    # Vars.
    use_saved_embeddings = True

    if use_saved_embeddings:
        X_train = joblib.load("X_train.pkl")
        X_test = joblib.load("X_test.pkl")
        y_train = joblib.load("y_train.pkl")
        y_test = joblib.load("y_test.pkl")
        tf = joblib.load("vectorizer.pkl")

    else:
        # Load corpus.
        print("load corpus")
        corpus_p = os.path.join("all-the-news", "articles_with_bias.csv")
        df = pd.read_csv(corpus_p)

        # Tf-idf embed the text.
        print("create embeddings")
        tf = TfidfVectorizer(max_features=5000, strip_accents="ascii") # implicitly normalizes to ASCII
        X = tf.fit_transform(df.content.values)

        # Encode the bias labels categorically.
        map = {
            "left bias" : 1,
            "left-center bias" : 2,
            "least biased" : 3,
            "right-center bias": 4,
            "right bias" : 5
        }
        y = [map[bias] for bias in df.bias]

        # Train/test split.
        print("train-test split")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # Save embeddings.
        print("save embeddings for next time.")
        arrs = [X_train, X_test, y_train, y_test, tf]
        fnames = ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl", "vectorizer.pkl"]
        for arr, fname in zip(arrs, fnames):
            joblib.dump(value=arr, filename=fname)


    """
    Train machine learning model.
    - l2 logistic regression model fit with 1-vs-rest.
    - TODO evaluate model.
    """

    # Train model.
    clf = LogisticRegression(penalty="l2", verbose=1)
    clf.fit(X_train, y_train)

    # Evaluate model: F score.
    test_ypred = clf.predict(X_test)
    f_score = f1_score(y_true=y_test, y_pred=test_ypred, average="micro")
    print("Test F1 = {:.4f}".format(f_score))

    # Evaluate model: confusion matrix.
    cm = confusion_matrix(y_true = y_test, y_pred=test_ypred)
    print(cm)


    """
    Save components to disk:
    - model.pkl
    - vectorizer.pkl
    """
    components = [clf, tf]
    fnames = ["model.pkl", "vectorizer.pkl"]
    for component, fname in zip(components, fnames):
        joblib.dump(value=component, filename=fname)


def make_bokeh_plot(bias_dict):
    """
    input: dictionary of bias values (e.g. "left bias") -> probabilities.
    output: tuple of
        resources:  css to support the bokeh plot
        script:     javascript for bokeh plot
        div:        container for bokeh plot.
    """

    # Express the data as a bokeh data source object.
    biases = ["left bias", "left-center bias", "least biased", "right-center bias",
              "right bias"]
    X = np.arange(5)
    y = [bias_dict[bias] for bias in biases]
    source = ColumnDataSource(data={
        "labels" : biases,
        "X" : X,
        "probabilities" : y
    })

    """ 
    Create plot. 
    """

    # Create plot object.
    width = 0.9
    p = figure(y_axis_label="Probability")
    p.vbar(source=source, x="X", top="probabilities", width=width)

    # Add hover tool, allowing mouseover to view text and sentiment.
    hover = HoverTool(
        tooltips=[
            ("probability", "@probabilities"),
        ],
        formatters={
            "text": "printf",
        },
        mode="vline"
    )
    p.add_tools(hover)

    """
    Format plot.
    """

    # axis font size
    p.xaxis.axis_label_text_font_size = "15pt"
    p.yaxis.axis_label_text_font_size = "15pt"

    # remove tick marks from axes
    p.xaxis.major_tick_line_color = None
    p.xaxis.minor_tick_line_color = None
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None

    # adjust plot width, height
    scale = 1.5
    p.plot_height = int(250 * scale)
    p.plot_width = int(450 * scale)

    # remove toolbar (e.g. move, resize, etc) from right of plot.
    p.toolbar.logo = None
    p.toolbar_location = None

    # remove gridlines
    p.xgrid.visible = False
    p.ygrid.visible = False

    # x axis ticks: use bias labels.
    xticks = np.arange(5)
    xtick_labels = ["Left Bias", "Left-Center\nBias", "Least Biased", "Right-Center\nBias",
              "Right Bias"]
    p.xaxis.major_label_overrides = {t : l for t, l in zip(xticks, xtick_labels)}

    """
    Export plot
    """

    # Create resources string, which is CSS, etc. that goes in the head of
    resources = INLINE.render()

    # Get javascript (script) and HTML div (div) for the plot.
    script, div = components(p)

    return (resources, script, div)


def generate_natural_language_explanation(bias_dict):
    """
    input: bias_dict mapping category (e.g. "left bias") to probability.
    output: string: natural language explanation like
        "This article likely has a conservative bias"
    """

    # First, calculate the overall bias as a string.
    bias_type = [k for k, v in bias_dict.items() if v == max(bias_dict.values())][0]

    # Next, calculate our confidence as string.
    confidence = ""
    threshold = 0.75
    if max(bias_dict.values()) >= threshold:
        confidence = "likely has"
    else:
        confidence = "may have"

    # Return a natural language explanation string.
    explanation = "This article {} a {}".format(confidence, bias_type)
    return explanation



    # Rule: if the dominant class is >= 75% probable, say likely.
    # Else: say may.



class BiasChecker:

    def __init__(self):
        """ load bias classifier and tokenizer"""
        self.clf = joblib.load("model.pkl")
        self.tf = joblib.load("vectorizer.pkl")

    def classify_bias(self, s):
        """
        input: article text as string.
        output: bias as dictionary mapping category -> probability
            e.g.,
            {
            "left-bias" : 0.2,
            "left-center-bias" : 0.15
            ....
            }
        """

        X = self.tf.transform([s])
        ypred = [math.e**v for v in self.clf.predict_log_proba(X)[0]] # list of probabilities
        biases = ["left bias", "left-center bias", "least biased", "right-center bias",
                  "right bias"]
        bias_dict = {bias : ypred_i for bias, ypred_i in zip(biases, ypred)}
        print(bias_dict)
        return bias_dict


class MainPage(webapp2.RequestHandler):

    def get(self):

        # Get text of the article from the website, entered by user.
        article_text = self.request.get("article_text")
        article_text = only_ascii(article_text).lower()

        # Classify bias of this article.
        b = BiasChecker()
        bias = b.classify_bias(article_text)

        # Produce bokeh plot of the probabilities.
        resources, script, div = make_bokeh_plot(bias_dict=bias)

        # Generate a natural language explanation of the result.
        explanation = generate_natural_language_explanation()

        # Load HTML template.
        html_p = os.path.join("html", "home.html")
        html = open(html_p, "r").read()

        # Replace placeholders in HTML with calculated values.
        placeholder_to_value = {
            "<!--bokeh_plot-->" : div,
            "<!--bokeh_resources-->" : resources,
            "<!--bokeh_script-->" : script,
            "<!--bias_explanation-->" : explanation
        }
        for placeholder, value in placeholder_to_value.ietms():
            html = html.replace(placeholder, value)

        # Write result as HTML output.
        self.response.headers["Content-Type"] = "text/html"
        self.response.write(html)


# Run application.
routes = [('/', MainPage)]
my_app = webapp2.WSGIApplication(routes, debug=True)

"""
### This main block can be used for quick offline app testing without the google app engine ###
if __name__ == "__main__":

    # Classify bias of this article.
    article_text = "This is some example article text which I think is just pretty neutral."
    b = BiasChecker()
    bias_dict = b.classify_bias(article_text) # e.g {"left" : 0.3}  map bias type to probability.

    # Produce bokeh plot of the probabilities.
    resources, script, div = make_bokeh_plot(bias_dict=bias_dict)

    # Generate a natural language explanation of the result.
    explanation = generate_natural_language_explanation(bias_dict=bias_dict)

    # Load HTML template.
    html_p = os.path.join("html", "home.html")
    html = open(html_p, "r").read()

    # Replace placeholders in HTML with calculated values.
    placeholder_to_value = {
        "<!--bokeh_plot-->": div,
        "<!--bokeh_resources-->": resources,
        "<!--bokeh_script-->": script,
        "<!--bias_explanation-->": explanation
    }
    for placeholder, value in placeholder_to_value.items():
        html = html.replace(placeholder, value)
        html = only_ascii(html)

    # Write test webpage.
    out_p = os.path.join("html", "result.html")
    with open(out_p, "w") as of:
        of.write(html)
        print("wrote to {}".format(out_p))
"""
