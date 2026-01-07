from flask import Flask, request,render_template
import torch
import json

app = Flask(__name__)



# ---------- LOAD MODEL ----------



with open("models/models.json", "r") as f:
    models = json.load(f)

for i in models["data"]:
    models["data"][i]["model"] = torch.jit.load(f"models/{i}_seq2seq.pt")





def vectToText(vect, int2phone):
    return " ".join([int2phone[int(i)] for i in vect if (i != 0 and i != 1)])


def str2data(text, model):
    
    max_length = model.decoder.max_length
    txt_split = text.strip().split(" ")
    phone2int = model.phone2int

    for i in txt_split:
        if not phone2int.get(i):
            return torch.zeros(max_length, dtype=torch.long)

    seq = [phone2int[i] for i in txt_split]+[phone2int["EOS"]]+[phone2int["PAD"]]*(max_length-len(txt_split))
    vect = torch.tensor(seq)

    return vect


def evaluate(encoder, decoder, sentences, int2phone):
    with torch.no_grad():
        input_tensor = torch.unsqueeze(sentences, 0)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = vectToText(decoded_ids, int2phone)

    return decoded_words, decoder_attn


# ---------- PAGES ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/description")
def description():
    return render_template("description.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/project", methods=["GET", "POST"])
def project_input():
    res = None
    text = ""
    selected_source = ""
    selected_target = ""
    phones = list()
    languages_selected = False
    source_option = list(models["abbr"].keys())
    target_option = list(models["abbr"].keys())
    
    if request.method == "POST":
        action = request.form.get("action")
        
        if action == "select_lang":
            selected_source = request.form.get("source_lang")
            selected_target = request.form.get("target_lang")
            languages_selected = True
            
            model_key = f"{selected_source}2{selected_target}"
            model_used = models["data"].get(model_key).get("model")
            phones = model_used.source_phone
        
        elif action == "run_model":
            selected_source = request.form.get("source_lang")
            selected_target = request.form.get("target_lang")
            text = request.form.get("texte", "")

            model_key = f"{selected_source}2{selected_target}"
            model_used = models["data"].get(model_key).get("model")
            model_used.eval()
            phones = model_used.source_phone
            languages_selected = True

            input_data = str2data(text.strip(" "), model_used)
            res = evaluate(
                model_used.encoder,
                model_used.decoder,
                input_data,
                model_used.int2phone
            )[0]

    return render_template(
        "project.html",
        source_option=source_option,
        target_option=target_option,
        phones=phones,
        languages_selected=languages_selected,
        selected_source=selected_source,
        selected_target=selected_target,
        res=res,
        texte=text
    )

if __name__ == "__main__":
    app.run(debug=True)
