import os, sys, codecs, random

'''
    This creates multi-classes training set from multi-files,
    Useage:
        python toCorpusAll.py inputdir outputdir key count
        class label are '0' and '1'
    output:
        train.dat test.dat
'''

inputdir = sys.argv[1]
outputdir = sys.argv[2]
savename = sys.argv[3]

corpus = {}
corpus_len = {}

filenames = os.listdir(inputdir)
filenames.sort()

for filename in filenames:
    _key = filename.strip().split(".")[0]
    if os.path.isdir(os.path.join(inputdir, filename)):continue
    datas = [x.strip() for x in codecs.open(os.path.join(inputdir, filename), "r", "gb18030")]
    corpus[_key] = datas
    corpus_len[_key] = len(datas)


print("Load Data Done!")
for _key in corpus:
    print("%s : %d" % (_key, corpus_len[_key]))

sout_config = open(os.path.join(outputdir, savename + ".label.config"), "w")
for label, filename in enumerate(filenames):
    _key = filename.strip().split(".")[0]
    sout_config.write(str(label) + ",1," + str(corpus_len[_key]) + "," + filename + "\n")
sout_config.close()

sout = codecs.open(os.path.join(outputdir, savename + ".dat"), "w", "gb18030")

for label, filename in enumerate(filenames):
    key = filename.strip().split(".")[0]
    for idx, data in enumerate(corpus[key]):
        sout.write(str(label) + "\t" + data + "\t" + data + "\n")

sout.close()

