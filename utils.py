
import os
import os.path
import torch

class saveData():
	def __init__(self, args):
		self.args = args
		self.save_dir = os.path.join(args.saveDir, args.load)
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		self.save_dir_model = os.path.join(self.save_dir, 'model')
		if not os.path.exists(self.save_dir_model):
			os.makedirs(self.save_dir_model)
		if os.path.exists(self.save_dir + '/log.txt'):
			self.logFile = open(self.save_dir + '/log.txt', 'a')
		else:
			self.logFile = open(self.save_dir + '/log.txt', 'w')		

	def save_model(self, model):
	    torch.save(
	        model.state_dict(),
	        self.save_dir_model + '/model_lastest.pt')
	    torch.save(
	        model,
	        self.save_dir_model + '/model_obj.pt')

	def save_log(self, log):
		self.logFile.write(log + '\n')

	def load_model(self, model):
		model.load_state_dict(torch.load(self.save_dir_model + '/model_lastest.pt'))
		print("load mode_status frmo {}/model_lastest.pt".format(self.save_dir_model))
		return model
