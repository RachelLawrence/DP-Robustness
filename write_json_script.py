import os
import subprocess
import tensorflow as tf

epsInputs = ['1', '2', '4'] 
numSamples = [100]

for num in numSamples:
	for eps in epsInputs:
		# If the save directory doesn't exist, create it
		saveDir = 'results/sample%d' % num
		if not os.path.isdir(saveDir):
			os.mkdir(saveDir)

		# If the save directory is empty, train the model and save it.
		cmd = ['python3', 'write_json_single.py']
		cmd.append(eps)
		cmd.append(str(num))

		subprocess.check_output(cmd)
		print('Results written for Eps %s, NumSamples %d' % (eps, num))