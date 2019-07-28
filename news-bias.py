"""
Virtual environment.
activate py27 (conda)
C:\users\hp user\news_app\scripts   --> type activate in shell.
"""

import pandas as pd
import numpy as np
import random
import os
import csv
import requests
import urllib
import json
import webapp2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.externals import joblib
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.embed import components, json_item
from bokeh.resources import INLINE
from bokeh.models.glyphs import Line, Text

def only_ascii(s):
    return "".join([c for c in list(s) if ord(c) <= 128])


def build_corpus(sources_per_class=50):
    """
    Build a corpus of news articles -- articles.csv -- labeled by their bias
    If corpus already exists, append the articles.

    Bias ratings based on bias_ratings.csv : we sample evenly from each of 5 classes
    (left bias .... --> right bias)

    We retrieve articles by querying the google news API for articles from each source.
    We represent an article by concatenating its title and text.
    Note that the text is often shortened by the google API.
    """

    # Load a table of sources and their biases.
    bias_df = pd.read_csv("bias_ratings.csv")

    # Open corpus output file. If it doesn't exist, make it and write a header.
    of_p = "articles.csv"
    if not os.path.exists(of_p):
        of = open(of_p, "wb")
        w = csv.writer(of)
        header = ["source", "bias", "text"]
        w.writerow(header)
        of.close()
    of = open(of_p, "ab")
    w = csv.writer(of)

    """
    # For each bias, randomly sample n_sources_per_class of these sources
    # And query the google news API for articles.
    # Write these articles to articles.csv
    """
    biases = ["left bias", "left-center bias", "least biased", "right-center bias",
              "right bias"]
    for bias in biases:

        # Select source_per_class random sources from this bias to sample.
        possible_sources = bias_df[bias_df.bias == bias].source.values
        shuffled_indices = np.arange(len(possible_sources))
        random.shuffle(shuffled_indices)
        sources = possible_sources[shuffled_indices][:sources_per_class]

        # Iterate over sources.
        for source in sources:

            # Get articles from google news API.
            API_KEY = "6aee96ef8754452ebfb3e8b1b33aec80"
            base_url = "https://newsapi.org/v2/everything"
            params = {
                "sources": source, # if multiple sources, pass comma-separated string.
                "apiKey": API_KEY
            }
            url = base_url + "?" + urllib.urlencode(params)


            try:
                result = requests.get(url)
                result_ascii = only_ascii(result.text)
                result_json = json.loads(result_ascii)["articles"]
            except Exception as e: # no articles for this source.
                print(e)
                result_json = []

            # Write output.
            for article_j in result_json:

                try:
                    out = [source, bias, article_j["title"] + " " + article_j["content"]]
                    w.writerow(out)
                except Exception as e: # e.g., empty article.
                    pass

    of.flush()
    of.close()


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

    # Load corpus.
    df = pd.read_csv("articles.csv")

    # Tf-idf embed the text.
    tf = TfidfVectorizer(max_features=5000)
    X = tf.fit_transform(df.text.values)

    # Encode the bias labels categorically.
    map = {
        "left bias" : 1,
        "left-center bias" : 2,
        "least biased" : 3,
        "right-center bias": 4,
        "right bias" : 5
    }
    y = [map[bias] for bias in df.bias]

    """
    Train machine learning model.
    - l2 logistic regression model fit with 1-vs-rest.
    - TODO evaluate model.
    """

    # Train model.
    clf = LogisticRegression(penalty="l2", verbose=1)
    clf.fit(X, y)

    # TODO Evaluate model.

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

    # remove x axis tick labels (done by setting label fontsize to 0 pt)
    p.xaxis.major_label_text_font_size = '0pt'

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
    explanation = "This article {} has a {} bias".format(confidence, bias_dict)
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

        X = self.tf.transform(s)
        ypred = self.clf.predict(X)

        return ypred  # TODO look up how to make ypred a dictionary not just most common label.


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