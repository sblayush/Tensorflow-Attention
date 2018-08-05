import os
import numpy as np
from math import exp
import json


def get_dir_path():
	return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))).replace('\\', '/')


def read_property_file(file_path):
	const_obj = {}
	with open(file_path, 'r') as f:
		for line in f.readlines():
			key, val = line.strip().split('=')
			const_obj[key] = val
	return const_obj


def check_if_model_run_already_there(model_path):
	if os.path.exists(model_path+'/checkpoint'):
		return True
	return False


def does_dir_exist(path):
	return os.path.exists(path)


def create_file(path):
	with open(path, 'w') as f:
		f.write('<html><head>'
				'<meta http-equiv="refresh" content="30">'
				'</head></html>')


def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def normalized1(x):
	a = [exp(i*5) for i in x]
	return a


def normalized2(x):
	a = [i*10 for i in x]
	return a


def normalized(x):
	# x = normalized1(x)
	norm1 = []
	mi = min(x)
	ma = max(x)
	for i in x:
		a = (i-mi)/(ma-mi)
		norm1.append(a)
	return norm1


def get_json_data(path):
	with open(path, 'r') as f:
		data = json.load(f)
	return data


def create_dirs(path):
	if not does_dir_exist(path):
		create_dir(path)


def save_feedback_data(new_data_list, path):
	if not does_dir_exist(path):
		create_dir(path)
	with open(path + str(date.today()).replace('-', '_') + '.json', 'w') as f:
		json.dump(new_data_list, f, indent=4)


def roundup(x):
	x += 200
	return x + 100 if x % 100 == 0 else x + 100 - x % 100


def write_json(obj, path, indent=2):
	with open(path, 'w') as f:
		json.dump(obj, f, indent=indent)


def read_json(path):
	with open(path) as f:
		d = json.load(f)
	return d


def print_progress_bar(iteration, total, prefix='Progress', suffix='Complete', decimals=2, length=50, fill='+'):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
		fill='█'
	"""
	iteration += 1
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filled_length = int(length * iteration // total)
	bar = fill * filled_length + '-' * (length - filled_length)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
	# Print New Line on Complete
	if iteration == total:
		print()


if __name__ == '__main__':
	from src.main.utility.get_logger import logger

	b = np.arange(1000)
	logger.info(str(normalized(b)))
	
	print(roundup(2123))
