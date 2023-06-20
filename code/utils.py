import tiktoken

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}. 
                                  See https://github.com/openai/openai-python/blob/main/chatml.md 
                                  for information on how messages are converted to tokens.""")


def ensure_fit_tokens(messages):
    """
    Ensure that total tokens in messages is less than MAX_TOKENS.
    If not, remove oldest messages until it fits.
    """
    total_tokens = num_tokens_from_messages(messages)
    while total_tokens > 4096:
        removed_message = messages.pop(0)
        total_tokens = num_tokens_from_messages(messages)
    return messages

def get_page_contents(docs):
    contents = ""
    metadata = ""
    for i, doc in enumerate(docs, 1):
        contents += f"Document #{i}:\n{doc.page_content}\n\n"
        metadata += f"Document #{i}\n{doc.metadata}\n\n"
    return contents, metadata

def extract_metadata(doc):
    cover = doc[0]
    text = cover.page_content

    while "  " in text:
        text = text.replace("  ", " ")
    text = text.replace(" \n", "\n")
    text = text.replace("\n", ". ")
    text = text.replace(" , ", ", ")
    text = text.replace("1 PONTIFICIA", "PONTIFICIA")

    while text[-1] == " ":
        text = text[:-1]
    
    return text

def add_info(doc):
    info = extract_metadata(doc)
    
    for page in doc:
        page.metadata.update({"info": info})
    
    return doc