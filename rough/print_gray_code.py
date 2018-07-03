def remove(arr):
	pass


def print_gray_code(arr1, arr2):
	print(arr1, arr2)
	if len(arr2) > 1:
		arr2 = arr2[1:]
		arr1 += arr2[:1]
		arr1, arr2 = print_gray_code(arr1, arr2)
	elif arr2[0] == 1:
		print_gray_code([], arr1 + [0])
	elif arr2[0] == 0:
		return arr1, arr2
	else:
		return arr1, [0]
	return arr1, arr2


if __name__ == '__main__':
	arr = [1, 1, 1, 1]
	a = print_gray_code([], arr)
	print(a)
