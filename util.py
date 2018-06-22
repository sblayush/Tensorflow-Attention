
def int_to_bin_list(num, length):
	return [pad([int(j) for j in list(bin(i))[2:]], length) for i in np.arange(num)]


def pad(arr, length):
	return np.pad(arr, (length-len(arr), 0), 'constant')
