import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.embed import components
from bokeh.resources import INLINE
import math
from flask import Flask, request

"""
Author: 
Robert Thorstad
thorstadrs@gmail.com
2019
MIT License.

---Overview: ------
This python script implements a news bias classifier in Flask, suitable for deployment as a web app.

When first executed, the classifier needs to be trained -- train_model().

After the model is trained, the index() function will generate a webpage to get the text of an article from a user,
and return HTML representing the model's classification of that article.

--- In more detail ---

At a high level, the logic is that many news sources have been well classified for their bias. Here we use ratings
from mediabiasfactcheck.com.

We access a large corpus of news articles from common sources (https://www.kaggle.com/snapcrack/all-the-news), and 
assume that, on average, an article from a source represents the bias of that source. 
"""

def only_ascii(s):
    """ helper function to convert string s to ascii. """
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

    # Optionally, use saved embeddings.
    # Large file, but saves several minutes.
    use_saved_embeddings = False

    # Using saved embeddings? Just load the pickled embeddings.
    if use_saved_embeddings:
        X_train = joblib.load("X_train.pkl")
        X_test = joblib.load("X_test.pkl")
        y_train = joblib.load("y_train.pkl")
        y_test = joblib.load("y_test.pkl")
        tf = joblib.load("vectorizer.pkl")

    # No saved embeddings? Make them.
    else:

        # Load corpus and lowercase.
        print("load corpus")
        corpus_p = os.path.join("all-the-news", "articles_with_bias.csv")
        df = pd.read_csv(corpus_p)
        df["content"] = [s.lower() for s in df.content]

        # Embed the text, using tf-idf vectorization.
        print("create embeddings")
        tf = TfidfVectorizer(max_features=5000, strip_accents="ascii")
        X = tf.fit_transform(df.content.values)

        # Map bias to a numerical label, e.g. "left bias" -> 1, "right bias" -> 5 , etc.
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

        # Randomly over-sample the training data until all classes are balanced.
        # This step could be omitted, saving time and memory, but would risk the classifier being
        # Biased to guess the more common classes.
        print("randomly undersample training data.")
        sampler = RandomOverSampler()
        X_train, y_train = sampler.fit_resample(X_train, y_train)

        # Save embeddings.
        print("save embeddings.")
        arrs = [X_train, X_test, y_train, y_test, tf]
        fnames = ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl", "vectorizer.pkl"]
        for arr, fname in zip(arrs, fnames):
            joblib.dump(value=arr, filename=fname)

    """
    Train machine learning model.
    """

    # Train model : currently we use L2-penalized MaxEnt (logistic regression)
    clf = LogisticRegression(penalty="l2", verbose=1)
    clf.fit(X_train, y_train)

    # Evaluate model: print an F1 score to screen.
    # Note: we use micro averaging to weight each class equally.
    # If we don't use oversampling, this is important to ensure the classifier performs well for all classes.
    test_ypred = clf.predict(X_test)
    f_score = f1_score(y_true=y_test, y_pred=test_ypred, average="micro")
    print("Test F1 = {:.4f}".format(f_score))

    # Print a confusion matrix to the screen.
    cm = confusion_matrix(y_true = y_test, y_pred=test_ypred)
    print(cm)

    """
    Save components to disk:
    - model.pkl     :       the maxent classifier 
    - vectorizer.pkl    :   tf-idf vectorizer
    """
    components = [clf, tf]
    fnames = ["model.pkl", "vectorizer.pkl"]
    for component, fname in zip(components, fnames):
        print("save {}".format(component))
        joblib.dump(value=component, filename=fname)


def make_bokeh_plot(bias_dict):
    """
    Generate an interactive plot representing the bias of an article.
    Input:
        bias_dict, which is a collection of bias -> probability mappings. E.g.:
        {
        "left bias" : 0.7,
        ...
        "right bias" : 0.2
        }
    Output:
        HTML, Javascript, CSS pieces necessary to render the graph, as strings.

        Tuple of:
        resources:  css to support the bokeh plot       as string
        script:     javascript for bokeh plot           as string
        div:        container for bokeh plot.           as string
    """

    # Express the data in a bokeh-compatible format, as a bokeh ColumnDataSource object.
    biases = ["left bias", "left-center bias", "least biased", "right-center bias",
              "right bias"]
    X = range(5)
    y = [bias_dict[bias] for bias in biases]
    source = ColumnDataSource(data={
        "labels" : biases,
        "X" : X,
        "probabilities" : y
    })

    """ 
    Create plot. 
    """

    # Create the initial plot object.
    width = 0.9
    p = figure(y_axis_label="Probability")
    p.vbar(source=source, x="X", top="probabilities", width=width)

    # Add Hover Tool, which allows mouseover to view classification probabilities.
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
    xticks = range(5)
    xtick_labels = ["Left Bias", "Left-Center\nBias", "Least Biased", "Right-Center\nBias",
              "Right Bias"]
    p.xaxis.major_label_overrides = {t : l for t, l in zip(xticks, xtick_labels)}

    """
    Export plot
    As, tuple of strings  (resources, script, div)
        resources :  CSS for the plot -- goes in HEAD
        script :     JavaScript for the plot -- goes in BODY
        div :        HTML container for the plot -- goes in BODY where the plot should appear
    """

    # CSS: resources.
    resources = INLINE.render() # string

    # Javascript (script) and HTML container (div)
    script, div = components(p) # string

    return (resources, script, div)

def generate_natural_language_explanation(bias_dict):
    """
    Given the output of the classifer, generate a natural language explanation like:
    "This article likely has a conservative bias"

    Input:
        bias_dict, which maps biases to probabilities e.g.:
            { "left bias" : 0.2 , etc }

    Output:
        a natural language explanation as string
    """

    # First, get the string label of the overall bias, defined as the most probable category.
    # E.g. "left-center bias"
    bias_type = [k for k, v in bias_dict.items() if v == max(bias_dict.values())][0]

    # Next, represent confidence as a string.
    # <= 75% confident in the label? --> "may have"
    # > 75% confident in the label? -> "likely has"
    confidence = ""
    threshold = 0.75
    if max(bias_dict.values()) >= threshold:
        confidence = "likely has"
    else:
        confidence = "may have"

    # Return a natural language explanation string.
    explanation = "This article {} a {}".format(confidence, bias_type)
    return explanation


class BiasChecker:
    """ Class to check the bias of an article, based on the trained machine learning model
        The most useful method is classify_bias(s) which takes the text of an article as string and returns
        the bias as a dictionary.
    """

    def __init__(self):
        """ Load bias classifier and tokenizer.
            These are serialized by train_model() which should be run first.
        """
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

        # Predict bias.
        # Generate a list of probabilities, where the indices map to
        # ["left bias", "left-center bias", "least biased", "right-center bias", "right bias"]
        X = self.tf.transform([s]).todense()
        ypred = [math.e**v for v in self.clf.predict_log_proba(X)[0]] # list of probabilities

        # Represent predicted bias as a dictionary mapping biases to probabilities.
        biases = ["left bias", "left-center bias", "least biased", "right-center bias",
                  "right bias"]
        bias_dict = {bias : ypred_i for bias, ypred_i in zip(biases, ypred)}

        return bias_dict


# By default, if an entrypoint is not defined in app.yaml,
# The Google Cloud engine looks in main.py for an object called app.
# Thus, initialize the app with this name.
app = Flask(__name__)

# This decorator binds a url -- here "/" -- to the method below.
# Thus, any time a user visits the root URL, index() will execute.
@app.route('/', methods=['GET', 'POST'])
def index():

    # Get text of the article, entered by the user.
    article_text = request.args.get("article_text")
    if not article_text: # no text entered? -> empty string.
        article_text = ""

    # Classify bias of this article.
    b = BiasChecker()
    bias = b.classify_bias(article_text)
    if article_text == "": # no article? Set all probabilities equal 0.2
        bias = { k : 0.2 for k in bias.keys() }

    # Produce bokeh plot of the probabilities.
    # The below are strings of CSS, JavaScript, and an HTML div respectively.
    resources, script, div = make_bokeh_plot(bias_dict=bias)

    # Generate a natural language explanation of the result.
    # Example: "This article may have a right-center bias."
    explanation = ""
    if article_text == "":
        explanation = "Please enter an article."
    else:
        explanation = generate_natural_language_explanation(bias_dict=bias)

    # Generate HTML, based on filling in a template with the graph and explanation generated.
    html_p = os.path.join("templates", "index.html")
    html = open(html_p, "r").read()
    placeholder_to_value = {
        "<!--bokeh_plot-->": div,
        "<!--bokeh_resources-->": resources,
        "<!--bokeh_script-->": script,
        "<!--bias_explanation-->": explanation
    }
    for placeholder, value in placeholder_to_value.items():
        html = html.replace(placeholder, value)
    return html

# If the app is run locally, this main block will execute.
if __name__ == '__main__':
    app.run(debug=True)
