#coding:utf8

import sys

if len(sys.argv) < 3:
	sys.exit(0)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(output_file,'w') as fp:
	for lines in open(input_file):
		lines = lines.strip().split('\t')
		data = float(lines[2])*100
		lines[2] = str(data)
		print >> fp,'\t'.join(lines)









