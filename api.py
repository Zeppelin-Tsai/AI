import configparser
import requests
import os

def get_kintone(url, app="", tokens=[], fields=[]):
    headers = {
        'X-Cybozu-API-Token': ','.join(tokens),  # 將 token 列表合併為一個字符串
        'Content-Type': 'application/json'
    }
    
    # 構建請求數據
    data = {
        "app": app if app else 289,  # 如果沒有提供 app，則使用預設的 289
        "fields": fields
    }
    
    try:
        # 發送 GET 請求
        response = requests.get(url, headers=headers, json=data)
        
        # 檢查狀態碼是否為 200，表示請求成功
        if response.status_code == 200:
            return response.json()  # 以 JSON 格式返回響應
        else:
            return {"error": f"Failed to fetch data, status code: {response.status_code}"}
    
    except Exception as e:
        return {"error": str(e)}  # 返回錯誤信息

# 範例調用
config = configparser.ConfigParser()

# 檢查 config.ini 檔案是否存在
config_file_path = './config.ini'
if not os.path.exists(config_file_path):
    print(f"Error: Configuration file '{config_file_path}' not found.")
    exit(1)

# 讀取設定檔
config.read(config_file_path)

# 確認 'API' 區段是否存在
if 'API' not in config:
    print("Error: 'API' section not found in the config file.")
    exit(1)

# 從設定檔中獲取 API 的配置信息
try:
    url = config['API']['url']
    app = config['API']['app']
    tokens = config['API']['tokens'].split(',')  # 分割 tokens 字符串
    
    # 嘗試讀取 'Fields' 區段，如果沒有則設置 fields 為空列表
    if 'Fields' in config and 'fields' in config['Fields']:
        fields = config['Fields']['fields'].split(',')  # 分割 fields 字符串
    else:
        fields = []  # 如果 'Fields' 不存在或 'fields' 鍵缺失，設置為空列表

except KeyError as e:
    print(f"Error: Missing expected key {str(e)} in the config file.")
    exit(1)
