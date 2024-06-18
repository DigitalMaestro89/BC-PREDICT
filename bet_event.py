import subprocess
from datetime import datetime
import json
import time

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
                "crash" : game_detail['rate']
            }
            history_array.append(new_item)
        return history_array[0]
    except subprocess.CalledProcessError as e:
        print(f"Error executing batch file: {e}")
    

def main() :
    latest_bet = get_history()
    print(latest_bet)
    while True:
        start_time = time.time()
        first_item = get_history()
        end_time = time.time()
        execution_time = end_time - start_time
        print(first_item)
        if latest_bet['gameId'] != first_item['gameId']:
            latest_bet = first_item
            print("Crash event occured")
            print(f"Time: {execution_time}")
            # print(latest_bet)
            print(f"Latest crash value: {latest_bet['crash']}")

if __name__ =="__main__":
    main()