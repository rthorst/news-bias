# set of small scripts to prepare news dataset.
# data from https://www.kaggle.com/snapcrack/all-the-news/downloads/all-the-news.zip/4
import pandas as pd

def add_bias_to_df():

    print("load articles")
    df = pd.read_csv("articles.csv")
    print(df.columns)
    print(set(df.publication))

    print("load biases")
    bias_df = pd.read_csv("biases.csv")
    source_to_bias = {s : b for s, b in zip(bias_df.source, bias_df.bias)}

    print("add bias to df")
    biases = [source_to_bias[source] for source in df.publication]
    df["bias"] = biases

    print("write output")
    df.to_csv("articles_with_bias.csv")

if __name__ == "__main__":
    add_bias_to_df()