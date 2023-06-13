import pandas as pd

df = pd.read_csv("output/metadata.csv", encoding="latin1")
df = df[["dc.contributor.advisor", "dc.contributor.author", "dc.date.issued", "dc.date.created", "dc.description.abstract", "dc.subject", "dc.title"]]
df.rename(columns = {"dc.contributor.advisor": "advisor", "dc.contributor.author":"author", "dc.date.issued":"date_issued", "dc.date.created":"date_created", 
                     "dc.description.abstract":"abstract", "dc.subject":"subject", "dc.title":"title"})


df.to_csv("output/data_clean.csv", encoding="latin1")