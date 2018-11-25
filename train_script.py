import os
import subprocess
import tensorflow as tf

# pcaInputs = [('no_pca', '0'), ('pca', '60')]
pcaInputs = [('pca', '60')]
epsInputs = ['0', '0.1', '0.2', '0.5', '1', '2', '4'] 
# epsInputs = ['0.1']
# numTrainingStepsInputs = ['600']
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
				cmd.append('--sigma=0')
				cmd.append('--num_training_steps=%s' % numTrainingSteps)
				cmd.append('--projection_dimensions=%s' % pca[1])
				cmd.append('--accountant_type=Amortized')

				subprocess.check_output(cmd)
				print('Checkpoint written to %s' % savePath)
			else:
				print("Checkpoint already exists in %s" % savePath)