# EconMentor

EconMentor is a chatbot web application based on the GPT-4 LLM created by OpenAI. Its current version uses a vector base hosted in Pinecone that contains abstracts and metadata from licensing thesis published in the Social Sciences faculty of the Pontifical Catholic University of Peru (PUCP). **Its goal is to aid prospective thesis writers in ideating new research topics.**

The following figure is a map of the different functions that make up EconMentor:

![code_map](readme/map.png)

The core function is `generate_response`, which handles the query and constructs the chain of messages to be fed to the model. It relies on two functions: first, `query_handler` performs a semantic search of the original user query through Maximum Marginal Relevance (MMR) to find the documents that score highest in terms of relevance and diversity ([Carbonell and Goldstein, 1998](https://dl.acm.org/doi/10.1145/290941.291025)). Then, `get_page_contents` parses the text and metadata of the found abstracts into a singular string. This string is then assigned as one value of a dictionary, the other being the role indicator, in this case `"user"`, as it contains the query and context provided to answer it. 

The second function `generate_response` relies on `construct_messages`, which generates the chain of messages to be processed by the model. This step is helpful as the GPT-4 API is, by default, best suited for chat completion; this means that it takes as context an initial system prompt and further messages both from the assistant itself and the user querying the model. This function mainly parses all the dictionaries, which contain each message and the role of the message's sender, into a list, all while assuring that the total number of tokens derived from the text is within the limits of what the model can handle (4096). 

Finally, `generate_response` sends the API request and obtains the response, which is then included as an HTML block to be displayed in the webpage created through the `streamlit` library. This latter library handles creating and customizing a container through which a chatbot webapp can be displayed and deployed.

## Local Use

To install and run the app locally, one can `clone` this repository and create a Python environment that includes the packages listed in [`requirements.txt`](requirements.txt). Then, you must include a file named `.env` with the following contents:

```
export OPENAI_API_KEY="<your openai api key>"
export PINECONE_API_KEY="<your pinecone api key>"
export PINECONE_ENVIRONMENT="<the pinecone environment in which your knowledgebase is hosted>"
```

Finally, in [`responseGeneration.py`](src/responseGeneration.py), change the `index_name` option for the name of your index. It might also be necessary to change the metadata fields indexed in the `get_page_contents` function included in the same file.

The app can be launched from the command line. While located in the root directory of the project and with the environment activated, one must run:

```
streamlit run src/app.py
```

## Demo Video

You can watch a short demo video for a previous version of the app [here](https://www.dropbox.com/scl/fi/dvl29re8irhfjpe80apzy/econMentor-v2.mp4?rlkey=vtzkb1jprirye80tgqloo4jtc&dl=0) (in Spanish).
