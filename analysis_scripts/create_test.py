import sys
import json 
from textblob import TextBlob
# test.jsonl
arg1 = sys.argv[1]
#arg2 = sys.argv[2]


test = open(arg1, "r").read().split("\n")
#model_output = open(arg2, "r").read().split("\n")


# test 1: drop all hindi words, test 2: drop all english, test 3: english on one side and hindi on another, test 4: text_en
f_1 = open("test_en.jsonl", "w")
f_2 = open("test_hi.jsonl", "w")
f_3 = open("test_order.jsonl", "w")
f_4 = open("test_translated.jsonl", "w")
f_5 = open("test_other.jsonl", "w")

for i in range(len(test)):
    if test[i] == "" or test[i] == " ":
        continue
    #val_1 = model_output[i].split("\t")[0].strip()
    val = json.loads(test[i])
    text = val["text_pp"]
    langids = val["langids_msftlid"]
    text_en = val["text_en"]
    text_1 = ""
    text_2 = ""
    text_3 = ""
    text_4 = ""
    text_5 = ""
    words = text.split(" ")
    lids = langids.split(" ")

    f = 0
    if len(words) != len(lids):
        f = 1

    for w, l in zip(words, lids):
        #print(w, l)
        #if len(w) < 3:
        #    continue
        #b = TextBlob(w)
        #l = b.detect_language()

        if l == "en":
            text_1 = text_1 + w + " "
        elif l == "hi":
            text_2 = text_2 + w + " "
        else:
            text_5 = text_5 + w + " "

    text_1 = text_1.strip()
    text_2 = text_2.strip()
    text_5 = text_5.strip()

    text_3 = text_1 + " " + text_2 + " " + text_5
    text_3 = text_3.strip()
    text_4 = text_en


    #print(text)
    #print(langids)
    #print(text_1)
    #print(text_2)
    #print(text_3)
    #print(text_4)
    #exit(0)
    if text_1 != "" and text_1 != " " and text_1 != "[]" and f == 0:
        val["text"] = text_1
        f_1.write(json.dumps(val))
        #f_1.write(str(val).replace("'", '"'))
        f_1.write("\n")
    if text_2 != "" and text_2 != " " and text_2 != "[]" and f == 0:
        val["text"] = text_2
        f_2.write(json.dumps(val))
        #f_2.write(str(val).replace("'", '"'))
        f_2.write("\n")

    if text_3 != "" and text_3 != " " and text_3 != "[]" and f == 0:
        val["text"] = text_3
        f_3.write(json.dumps(val))
        #f_3.write(str(val).replace("'", '"'))
        f_3.write("\n")

    if text_4 != "" and text_4 != " " and text_4 != "[]":
        val["text"] = text_4
        f_4.write(json.dumps(val))
        #f_4.write(str(val).replace("'", '"'))
        f_4.write("\n")

    if text_5 != "" and text_5 != " " and text_5 != "[]" and f == 0:
        val["text"] = text_5
        f_5.write(json.dumps(val))
        f_5.write("\n")


