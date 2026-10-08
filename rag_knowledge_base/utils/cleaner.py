import re

def clean_text(text):
    text = text.replace("\r\n","\n")
    text = text.replace("\t"," ")
    text = re.sub(r"[ ]+"," ",text)
    text = re.sub(r"\n{3,}","\n\n",text)
    text = text.strip()
    return text