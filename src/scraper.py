import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import warnings

# def main():
#     links = []
#     master = pd.DataFrame()
#     warnings.filterwarnings("ignore")

#     for i in np.arange(15):
#         url = f"https://tesis.pucp.edu.pe/repositorio/handle/20.500.12404/1026/recent-submissions?offset={i * 20}"
#         response = requests.get(url, verify=False) # get request to url
#         soup = BeautifulSoup(response.content, "html.parser") # parse with bs
#         item_elements = soup.find_all("h4", class_="artifact-title") # find all title items
        
#         for item in item_elements:
#             link = "https://tesis.pucp.edu.pe" + item.find("a")["href"] + "?show=full" # get all links to metadata for each title
#             links += [link]
#     print(f"Going through all {len(links)} links. This might take a while")
#     print(" ----+---- ----+---- ----+---- ----+---- ----+---- ")

#     for index, link in enumerate(links):
#         print(".", end="")

#         if (index + 1) % 50 == 0:
#             print("    ", index + 1, sep="")
#         response = requests.get(link, verify=False)
#         soup = BeautifulSoup(response.content, "html.parser")
#         table = str(soup.find("table"))
#         df = pd.read_html(str(table))[0][[0, 1]]
#         df = df.groupby(0, as_index=False).agg("\n".join)
#         df = df.set_index(0).T.reset_index().drop(columns=["index"])
#         master = pd.concat([master, df])
    
#     master.to_csv("output/metadata.csv", encoding="latin1", errors="replace")

thesisListPageLinkBase = "https://tesis.pucp.edu.pe/repositorio/handle/20.500.12404/1026/recent-submissions?offset={offset}"



if __name__ == "__main__":
    main()
