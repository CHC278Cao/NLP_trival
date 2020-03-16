
import os
import pdb
import docopt
import numpy as np
import pandas as pd

from . import tfIdf
from . import wordCount


def main():
    trainfile = "./inputs/train.csv"
    validfile = "./inputs/valid.csv"
    testfile = "./inputs/test.csv"

    train_df = pd.read_csv(trainfile).reset_index(drop=True)
    valid_df = pd.read_csv(validfile).reset_index(drop=True)
    test_df = pd.read_csv(testfile).reset_index(drop=True)

    train_df = train_df.sample(frac=1, random_state=42)
    valid_df = valid_df.sample(frac=1, random_state=42)
    print(train_df.head())
    print(valid_df.head())

    tfIdf.train(train_df, valid_df, test_df)


if __name__ == "__main__":
    main()