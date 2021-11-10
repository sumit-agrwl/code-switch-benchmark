import sys

# script compares failures between these files
arg1 = sys.argv[1] # normal

arg2 = sys.argv[2] # task adaptive

arg3 = sys.argv[3]

arg4 = sys.argv[4]

# 1_pass_2fail
# 1_fail_2pass

f_1 = open(arg3, "w")
f_2 = open(arg4, "w")

data1 = open(arg1, "r").read().split("\n")
data2 = open(arg2, "r").read().split("\n")

for d1, d2 in zip(data1, data2):
    if d1 == "" or d1 == " ":
        continue

    val1 = d1.split("\t")
    val2 = d2.split("\t")

    text_1 = val1[0]
    t1 = val1[2]
    p1 = val1[4]

    text_2 = val2[0]
    t2 = val2[2]
    p2 = val2[4]

    #print(text_1, t1, p1)
    #print(text_2, t2, p2)

    f1 = 0
    f2 = 0

    if t1 == p1:
        f1 = 1 # pass
    if t2 == p2:
        f2 = 1

    if f1 == 1 and f2 == 0:
        f_1.write(text_1 + "\t" + t1 + "\n")
    if f1 == 0 and f2 == 1:
        f_2.write(text_1 + "\t" + t1 + "\n")
