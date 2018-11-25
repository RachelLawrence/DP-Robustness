import os
import subprocess
import tensorflow as tf

pcaInputs = [('no_pca', '0'), ('pca', '60')]
epsInputs = ['0'] #, '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
				# ['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09']
numTrainingStepsInputs = ['10']

for pca in pcaInputs:
	for eps in epsInputs:
		for numTrainingSteps in numTrainingStepsInputs:
			# If the save directory doesn't exist, create it
			savePath = 'trained_models/%s_eps_%s_%s' % (pca[0], eps, numTrainingSteps)
			if not os.path.isdir(savePath):
				os.mkdir(savePath)

			# If the save directory is empty, train the model and save it.
			if not os.path.isfile("%s/checkpoint" % savePath):
				cmd = ['python3', 'third_party/differential_privacy/dp_sgd/dp_mnist/dp_mnist.py']
				cmd.append('--training_data_path=dp_data/mnist_train.tfrecord')
				cmd.append('--eval_data_path=dp_data/mnist_test.tfrecord')
				cmd.append('--save_path=%s' % savePath)
				cmd.append('eps=%s' % eps)
				cmd.append('--num_training_steps=%s' % numTrainingSteps)
				cmd.append('--projection_dimensions=%s' % pca[1])

				subprocess.check_output(cmd)
				print('Checkpoint written to %s' % savePath)
			else:
				print("Checkpoint already exists in %s" % savePath)