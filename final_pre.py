# -*- encoding:utf-8 -*-
import codecs
import re
f1_lines = codecs.open("resultWikiQA20").readlines()
f2_lines = codecs.open("../checkpoint_transformer_classification/result_transformer_classification.csv").readlines()
flabel_lines = codecs.open("../data/wikiqa-test-y.txt").readlines()
total = len(f1_lines)
qids = []
score1s = []
score2s = []
labels = []
for i in range(total):
	l1 = f1_lines[i].strip()
	qid, s1, s2, score1 = l1.split("\t")
	score1 = float(score1[1: len(score1) - 1])
	score1s.append(score1)
	qids.append(qid)
	l2 = f2_lines[i].strip()
	l2 = re.split("\s+", l2[1: len(l2) - 1])
	#score2 = float(l2[1]) - float(l2[0])
	score2 = float(l2[1])
	score2s.append(score2)
	labels.append(int(flabel_lines[i].strip()))

answers = {}
MAP, MRR = 0, 0
for i in range(total):
	if qids[i] in answers:
		answers[qids[i]].append((score1s[i] + score2s[i], labels[i]))
	else:
		answers[qids[i]] = [(score1s[i] + score2s[i], labels[i])]

for i in answers.keys():
	p, AP = 0, 0
	MRR_check = False
	answers[i] = sorted(answers[i], key=lambda x: x[0], reverse=True)
	print(answers[i])
	for idx, (s, l) in enumerate(answers[i]):
		if l == 1:
			if not MRR_check:
				MRR += 1 / (idx + 1)
				MRR_check = True
			p += 1
			AP += p / (idx + 1)
	if p == 0:
		print(answers[i])
	AP /= p
	MAP += AP

total_q = len(answers.keys())
MAP /= total_q
MRR /= total_q
print("MAP", MAP, "MRR,", MRR)

