import pandas as pd
import os

dataFilePath = "data"
pdfFilesPath = f"{dataFilePath}/3_licenciatura/facultad_de_ciencias_sociales/economia_(lic)/"
pdfFilesPathInMetadata = pdfFilesPath.replace(dataFilePath, "dspace_home")
data = pd.read_json("data/00_metadata/eco_tesis.json")
pdfFileNames = os.listdir(pdfFilesPath)
pdfFileLocalData = data.loc[data["pdf_avaible_online"], "pdf_file_local"].str.replace(pdfFilesPathInMetadata, "")
matchingPDFs = list(set(pdfFileNames).intersection(set(pdfFileLocalData)))