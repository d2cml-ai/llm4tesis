import pandas as pd

df = pd.read_csv("output/metadata.csv", encoding="latin1")

cols = [
    "dc.contributor.advisor", "dc.contributor.author", "dc.date.issued", 
    "dc.date.created", "dc.description.abstract", "dc.subject", "dc.title", 
    "renati.type"
]
df = df[cols]
df = df.rename(columns = {
    "dc.contributor.advisor": "advisor", "dc.contributor.author": "author", 
    "dc.date.issued": "date_issued", "dc.date.created": "date_created", 
    "dc.description.abstract": "abstract", "dc.subject": "subject", 
    "dc.title":"title", "renati.type": "type"
})
df = df.query("type != 'https://purl.org/pe-repo/renati/type#trabajoDeSuficienciaProfesional'")

df.to_csv("output/tesis_data.csv", encoding="latin1")

