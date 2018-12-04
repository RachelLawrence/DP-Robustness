import os
import subprocess
import tensorflow as tf

#pcaInputs = [('jl', '60'), ('pca', '60'), ('no_pca', '0')]
pcaInputs = [('ndppca', '60')]

epsInputs = ['0', '0.5', '1', '1.5', '2', '3', '4'] 
numTrainingStepsInputs = ['600']

for pca in pcaInputs:
	for eps in epsInputs:
		for numTrainingSteps in numTrainingStepsInputs:
			# If the save directory doesn't exist, create it
			savePath = 'trained_models/%s_eps_%s_%s_amortized' % (pca[0], eps, numTrainingSteps)
			if not os.path.isdir(savePath):
				os.mkdir(savePath)

			# If the save directory is empty, train the model and save it.
			if not os.path.isfile("%s/checkpoint" % savePath):
				cmd = ['python3', 'third_party/differential_privacy/dp_sgd/dp_mnist/dp_mnist.py']
				cmd.append('--training_data_path=dp_data/mnist_train.tfrecord')
				cmd.append('--eval_data_path=dp_data/mnist_test.tfrecord')
				cmd.append('--save_path=%s' % savePath)
				cmd.append('--eps=%s' % eps)
				cmd.append('--end_eps=%s' % eps)
				cmd.append('--num_training_steps=%s' % numTrainingSteps)
				cmd.append('--projection_dimensions=%s' % pca[1])
				cmd.append('--accountant_type=Amortized')

				if pca[0] is not 'no_pca':
					cmd.append('--projection_type=%s' % pca[0].upper())


				subprocess.check_output(cmd)
				print('Checkpoint written to %s' % savePath)
			else:
				print("Checkpoint already exists in %s" % savePath)