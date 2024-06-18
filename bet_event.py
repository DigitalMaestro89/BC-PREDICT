import subprocess
from datetime import datetime
import json
import time
import hmac
import hashlib

def get_history():
    batch_file_path = "site.bat"
    history_array = []
    try:
        result = subprocess.run(batch_file_path, capture_output=True, text=True, check=True)
        stdout = result.stdout
        outputs = stdout.split('\n')
        result = json.loads(outputs[-1])
        data = result['data']

        for item in data['list']:
            game_detail = json.loads(item['gameDetail'])
            game_detail['prepareTime'] = datetime.fromtimestamp(game_detail['prepareTime'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            game_detail['beginTime'] = datetime.fromtimestamp(game_detail['beginTime'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            game_detail['endTime'] = datetime.fromtimestamp(game_detail['endTime'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            item['gameDetail'] = game_detail
            new_item = {
                "gameId" : item["gameId"],
                "hash" : game_detail["hash"],
                "crash" : game_detail['rate'],
                "salt" : game_detail['salt']
            }
            history_array.append(new_item)
        return history_array[0]
    except subprocess.CalledProcessError as e:
        print(f"Error executing batch file: {e}")

# get crash value from its hash value
def get_crash_from_hash(hash, salt):
	hash = hmac.new(salt.encode(), bytes.fromhex(hash), hashlib.sha256).hexdigest()
	n_bits = 52
	r = int(hash[:n_bits // 4], 16)
	X = r / (2 ** n_bits)
	X = float(f'{X:.9f}')
	X = 99 / (1 - X)
	result = int(X)
	return max(1, result / 100)

# get precious hash value from next hash
def get_previous_hash(hash):
	return hashlib.sha256(hash.encode()).hexdigest()

# get latest 200 crash history for predicting
def get_latest_history(latest_hash, game_id, salt):
    end_point = game_id - 200
    # get crash history
    crash_history = []
    while game_id > end_point:
       crash = get_crash_from_hash(latest_hash, salt)
       crash_history.append(crash)
       game_id -= 1
       latest_hash = get_previous_hash(latest_hash)
    # reverse the history array
    crash_history = crash_history[::-1]
    return crash_history

# execute when the crash event occured
def auto_bet(latest_hash, game_id, salt):
    # get latest 200 crash values
    history = get_latest_history(latest_hash, game_id, salt)
    
    return 0

def main() :
    # get the latest crash event
    latest_bet = get_history()
    print(latest_bet)
    # watch when the next crash event occur
    while True:
        first_item = get_history()
        print(first_item)
        if latest_bet['gameId'] != first_item['gameId']:
            latest_bet = first_item
            print("Crash event occured")
            print(f"Latest crash value: {latest_bet['crash']}")
            auto_bet(latest_bet['hash'], latest_bet['gameId'], latest_bet['salt'])            


if __name__ =="__main__":
    main()