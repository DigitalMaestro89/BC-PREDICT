import hmac
import hashlib
import json

def get_previous_hash(hash):
	return hashlib.sha256(hash.encode()).hexdigest()

def get_crash_from_hash(hash, salt):
	hash = hmac.new(salt.encode(), bytes.fromhex(hash), hashlib.sha256).hexdigest()
	n_bits = 52
	r = int(hash[:n_bits // 4], 16)
	X = r / (2 ** n_bits)
	X = float(f'{X:.9f}')
	X = 99 / (1 - X)
	result = int(X)
	return max(1, result / 100)

def build_database(index, hash, salt):
	# for i in range(0, 10, 1):
	end_point = index - 3500
	data = []
	while index > end_point:
		crash_value = get_crash_from_hash(hash, salt)
		data.append(crash_value)
		index -= 1
		hash = get_previous_hash(hash)

	data.reverse()
	file = f"history{i}.json"
	with open(file, "w") as file:
		file.write(json.dumps(data, indent=2))
	file.close()

	print('Build history data')
	return 0

def main():
	salt = '0000000000000000000301e2801a9a9598bfb114e574a91a887f2132f33047e6'
	index = 7100871
	hash = 'aab133c16cc62f1ae25017d4c7793d1c1111c08437a83323986723eeaafdc7dc'
	build_database(index, hash, salt)

main()