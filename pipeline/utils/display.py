from spacy import displacy

def display_ent(doc, style="ent", colors=None, options=None, compact=True):
    colors = colors or {"TRIGGER": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    options = options or {"ents": None, "colors": colors, "compact": compact}
    displacy.render(doc, style=style, jupyter=True, options=options)