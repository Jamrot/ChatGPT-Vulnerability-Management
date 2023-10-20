import tiktoken

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def message_process(message, max_token=32000, stopwords_process=0):
    """delete the part longer than max token in the message (modified from num_tokens_from_messages)"""
    
    # token_num = num_tokens_from_messages(message)
    stop_num = max_token - 2
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    num_tokens += 4
    for key, value in message.items():
        if stopwords_process: 
            value = stopwords_process(value)
        value_split = value.split(" ")
        message_content = ""
        for word in value_split:
            num_tokens += len(encoding.encode(word))          
            if not num_tokens>=stop_num:
                message_content = " ".join([message_content, word])
            else:
                break
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2
    message = {"role": "user", "content": message_content}
    
    return message

# if __name__=="__main__":
#     print(num_tokens_from_messages([{"role":"user", "content":""}]))