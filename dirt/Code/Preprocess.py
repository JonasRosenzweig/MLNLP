# Make a script that can make sure we have same amount of negative and positive rows.
import os
import pandas as pd

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

filename = os.listdir("C:\\Users\\HE400\\PycharmProjects\\MLNLP2\\Tutorials_and_References\\notebookGold\\input")[0]

dataset_path = os.path.join("..",
                            "C:\\Users\\HE400\\PycharmProjects\\MLNLP2\\Tutorials_and_References\\notebookGold\\input",
                            filename)

df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)


df.loc[df['target'] == '']
