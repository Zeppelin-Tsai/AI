import os
import json
import sqlite3
import pandas as pd
import openai
import hashlib
import logging
from api import get_kintone
import configparser



# 手动创建文件处理器并设置编码为 utf-8
file_handler = logging.FileHandler('app.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 设置日志记录功能
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# 示例: 记录某个文件的处理信息
logger.info(f"日志初始化成功，开始记录操作信息。")

# 设置资料夹路径
DOCS_FOLDER = './docs'
OUTPUT_FOLDER = './output'
DB_FILE = 'data.db'  # SQLite 数据库文件
METADATA_FILE = os.path.join(OUTPUT_FOLDER, 'metadata.json')
OUTLINE_CACHE_FILE = os.path.join(OUTPUT_FOLDER, 'outline_cache.txt')

# 加载本地快取的大綱
def load_outline_cache():
    if os.path.exists(OUTLINE_CACHE_FILE):
        with open(OUTLINE_CACHE_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return None

# 保存大綱到本地快取
def save_outline_cache(outline):
    with open(OUTLINE_CACHE_FILE, 'w', encoding='utf-8') as f:
        f.write(outline)
        
def setup_logging(debug_mode=False):
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
# 读取配置文件
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

# 设置 OpenAI API 密钥
config = load_config()
openai.api_key = config.get('OPENAI_API_KEY')



# 创建输出资料夹（如果不存在）
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
# 读取 Kintone 数据并保存
def fetch_and_save_kintone_data():
    # 从 Kintone 获取数据
    logger.info("从 Kintone 获取数据...")
    
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
        
    kintone_data = get_kintone(url, app, tokens, fields)
    
    if "error" in kintone_data:
        logger.error(f"获取 Kintone 数据时发生错误: {kintone_data['error']}")
        return None
    return kintone_data
# 读取元数据
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 保存元数据
def save_metadata(metadata):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

# 获取文件元数据（哈希值和修改时间）
def get_file_metadata(filepath):
    metadata = {}
    metadata['size'] = os.path.getsize(filepath)  # 获取文件大小
    metadata['last_modified'] = os.path.getmtime(filepath)  # 获取文件最后修改时间
    metadata['hash'] = hash_file(filepath)  # 计算哈希值（內容變化）
    return metadata# 計算檔案的哈希值

def hash_file(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# 檢查檔案是否更新（優先檢查文件大小和修改時間）
def file_updated(filepath, previous_metadata):
    current_metadata = get_file_metadata(filepath)
    
    if previous_metadata is None:
        return True

    # 先檢查文件大小
    if current_metadata['size'] != previous_metadata.get('size'):
        logging.info(f"文件大小不同，{filepath} 有變動。")
        return True
    
    # 再檢查最後修改時間
    if current_metadata['last_modified'] != previous_metadata.get('last_modified'):
        logging.info(f"最后修改时间不同，{filepath} 有變動。")
        return True
    
    # 最後才檢查哈希值
    if current_metadata['hash'] != previous_metadata.get('hash'):
        logging.info(f"哈希值不同，{filepath} 有變動。")
        return True

    return False

# 将 CSV 文件转换为 SQLite 数据库
def csv_to_sqlite(csv_file, db_file):
    conn = sqlite3.connect(db_file)
    df = pd.read_csv(csv_file)
    # 格式化表名，移除 .csv 後綴和特殊字符
    table_name = os.path.splitext(os.path.basename(csv_file))[0].replace('-', '_').replace('.', '_')
    # 列名去除空格和特殊字符
    df.columns = [col.replace(' ', '_').replace('-', '_').replace('.', '_') for col in df.columns]
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"文件 {csv_file} 成功导入到 SQLite 数据库 {db_file} 中的表 {table_name}")
# 输入验证函数，检查问题是否有效
def validate_input(question):
    if not question.strip():
        logging.warning("用户输入为空。")
        return False
    if len(question) < 5:
        logging.warning(f"问题太短，可能无效：{question}")
        return False
    return True

# 1.建立資料索引、大綱並轉換CSV
def build_data_index_and_convert(docs_folder, db_file, force_update=False):
    index = {}
    outline = ""
    metadata = load_metadata()
    metadata_updated = False  # 用於判斷是否需要更新 metadata
    
    # 获取 Kintone 数据
    kintone_data = fetch_and_save_kintone_data()
    if not kintone_data:
        print("未获取到 Kintone 数据，请检查 API 配置。")
        return index, outline

    # 处理 Kintone 数据
    logger.info("处理 Kintone 数据...")
    fields_to_remove = ["$revision", "建立人", "更新人", "更新時間", "$id", "建立時間"]
    
    if "records" in kintone_data:
        cleaned_records = []
        for record in kintone_data['records']:
            cleaned_record = {k: v['value'] if isinstance(v, dict) and 'value' in v else v
                              for k, v in record.items() if k not in fields_to_remove}
            cleaned_records.append(cleaned_record)
        
        kintone_df = pd.DataFrame(cleaned_records)
        kintone_columns = kintone_df.columns.tolist()
        index['kintone_data'] = {'columns': kintone_columns}
        
        new_outline = f"档案名称：kintone_data\n栏位名称：{', '.join(kintone_columns)}\n前五笔资料样本：\n"
        sample_data = kintone_df.head(5).to_dict(orient='records')
        for record in sample_data:
            new_outline += ', '.join([f"{k}: {v}" for k, v in record.items()]) + "\n"
        outline += new_outline + "\n"

        conn = sqlite3.connect(db_file)
        kintone_df.to_sql('kintone_data', conn, if_exists='replace', index=False)
        conn.close()
        logger.info("Kintone 数据成功导入到 SQLite 表 'kintone_data'")
    
    # 檢查 docs_folder 中的文件
    print(f"检查 {docs_folder} 文件夹中的文件...")
    for filename in os.listdir(docs_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(docs_folder, filename)
            prev_metadata = metadata.get(filename)

            if not force_update and not file_updated(file_path, prev_metadata):
                print(f"文件 {filename} 没有更新，跳过。")
            else:
                print(f"文件 {filename} 有变动，读取文件并更新数据库: {file_path}")
                csv_to_sqlite(file_path, db_file)
                metadata[filename] = get_file_metadata(file_path)
                metadata_updated = True

            df = pd.read_csv(file_path, dtype=str, encoding='utf-8')
            columns = df.columns.tolist()
            index[filename] = {'columns': columns}

            new_outline = f"档案名称：{filename}\n栏位名称：{', '.join(columns)}\n前五笔资料样本：\n"
            sample_data = df.head(5).to_dict(orient='records')
            for record in sample_data:
                new_outline += ', '.join([f"{k}: {v}" for k, v in record.items()]) + "\n"
            outline += new_outline + "\n"

    # 如果有更新元数据，保存
    if metadata_updated:
        save_metadata(metadata)

    # 保存大綱快取
    save_outline_cache(outline)
    
    # 如果未生成资料索引或大纲，强制生成一份空的
    if not index and not outline:
        print("未生成資料索引或大綱，強制生成一個空的結果。")
        outline = "未生成有效大綱。"
    
    return index, outline

# 保存用户问题和SQL查询的上下文
context = []
query_cache = {}
# 2生成 SQL 查询
def generate_sql_query(outline, question, context):
    if question in query_cache:
        logging.info(f"从缓存中返回查询结果：{question}")
        return query_cache[question], None  # 返回缓存的查询
    
    previous_queries = "\n".join([f"问题: {q['question']}\nSQL: {q['sql']}\n结果: {q['result']}" for q in context])

    # 修改提示，要求 GPT 返回详细的错误信息
    prompt = f"""
    以下是数据文件的摘要：
    {outline}

    之前的问题和查询结果：
    {previous_queries}

    用户的当前问题是：{question}

    请生成一条有效的 SQL 查询语句，符合 SQLite 语法规则。
    注意事项：
    1. 表名中(同时为文件名) **不要包含文件扩展名**，如 .csv。
    2. 表名或列名中的空格、破折号等特殊字符应替换为下划线 _。
       但是 values 中的数据不需要替换。
    3. 请确保 SQL 查询中使用正确的逻辑运算符，如 AND、OR 等。
    4. 对于可能涉及姓名、地址或其他类似字段的搜索，请确保支持模糊查找，并考虑以下情况：
       - 简体中文与繁体中文的匹配；
       - 日文姓名的匹配；
       - 全名或部分名；
       - 有或没有空格的姓名，如“愛媛愛媛” 和 “愛媛 愛媛” 都应该匹配。
    5. 如果查询包含模糊搜索，使用 `LIKE` 或 `ILIKE` 语句。
    6. 如果查询需要唯一的结果，请使用 `DISTINCT`。
    7. 查询应包括所有表，并使用 `UNION` 操作符将多个表的数据组合在一起。
    8. 如果查询可能返回大量行，请考虑使用 `LIMIT` 限制结果数量。
    9. 如果查询结果需要排序，请使用 `ORDER BY`。
    10. 如果查询需要多个表，请使用 `JOIN` 或 `UNION`。
        重要提示：盡可能地查過所有有可能性的表，並使用 UNION 將所有結果合併。
        如SELECT * FROM sample_29524 WHERE 姓_名 LIKE '%nanako%' UNION SELECT * FROM sample_500 WHERE 姓_名 LIKE '%nanako%' UNION SELECT * FROM kintone_data WHERE 姓_名 LIKE '%nanako%';
    11. 如果查询需要计算，请使用 `SUM`、`AVG`、`COUNT` 等聚合函数。
    12. 如果查询需要日期范围，请使用 `BETWEEN`。
    13. 对于涉及时间字段的查询，如工数，请动态解析 'hh:mm' 格式为小时数。
        範例：SELECT SUM(
        CAST(SUBSTR(工数, 1, INSTR(工数, ':') - 1) AS INTEGER) +  -- 提取小时部分
        CCAST(SUBSTR(工数, INSTR(工数, ':') + 1) AS INTEGER) / 60.0  -- 提取分钟部分并转换为小时
        C) AS 工数总计 
        CFROM sample_500 
        CWHERE 姓_名 LIKE '%広島 広島%';
    14. 当用户问题涉及到列出所有种类或详细信息时，请确保查询列出所有相关项而不仅仅返回总数。示例：如果用户问到「有多少种项目名」，SQL 应该列出所有项目名。
    15. 对于可能的 SQL 注入攻击，请确保查询是安全的。
    16. 如果查询需要使用变量，请使用 `?` 占位符。
    17. 送之前嚴格检查SQL查询語句中的欄位是否存在于给定的数据文件中，
        從檔案大綱可以判斷。
        如果某个欄位不存在（如平均工资或年齡），
        或是查詢的情況不符合邏輯（如查詢平均工资(金錢)，卻用工数(時間)當計算），
        请返回详细的错误信息。
    18. 请严格按照以下格式返回响应
        (SQL跟ERROR只能擇一回報，模糊判斷，SQL優先)
        不要同時傳SQL跟ERROR(好比ERROR: 无错误。)
        甚至是其他內容，否則將無法正確解析。
        
        SQL: <生成的查询语句>
        
        或是(只能擇一，SQL優先)
        
        ERROR: <如果字段不存在或查询无效，提供详细错误描述>
       
    範例SQL: SELECT 姓_名 FROM sample_29524 WHERE スタッフコード = 42 UNION SELECT 姓_名 FROM sample_500 WHERE スタッフコード = 42;
    範例ERROR: 沒有跟年齡相關的欄位，無法進行查詢。
    """

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0,
        max_tokens=500,
    )
    
    gpt_response = response.choices[0].message['content'].strip()

    # 输出 GPT 的完整响应用于调试
    print("GPT 响应完整内容: ")
    print(gpt_response)

    # 处理错误响应的逻辑
    if "ERROR:" in gpt_response:
        error_message = gpt_response.split("ERROR:")[1].strip()  # 提取详细错误信息
        print(f"错误: {error_message}")
        return None, error_message  # 返回 None 和详细的错误信息

    # 正常情况下处理SQL查询
    if "SQL:" in gpt_response:
        sql_query = gpt_response.split("SQL:")[1].strip()
        query_cache[question] = sql_query  # 缓存查询结果
        return sql_query, None  # 返回SQL查询和无错误
    else:
        print("错误: 未找到正确格式的 SQL 响应。")
        return None, "未找到正确格式的 SQL 响应"


# 在执行查询后发送结果到 GPT
def send_results_to_gpt(question, sql_query, result):
    prompt = f"""
    这是用户的问题和部分查询结果：
    问题: {question}
    SQL查询: {sql_query}
    查询结果: (前100行)
    {result[:100]}  # 如果查询结果过长，可以只传递前100行
    
    请根据以上信息提供详细的答案或进一步的建议。
    请根据用户提问的语言回答，使用相应的语言（简体中文、繁体中文或日文）。
    """
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0,
        max_tokens=500,
    )
    gpt_response = response.choices[0].message['content'].strip()
    print("GPT 针对查询结果的响应: ")
    print(gpt_response)


# 执行 SQL 查询，使用参数化查询防止 SQL 注入
def execute_sql_query(db_file, sql_query, question, context):
    if not sql_query:
        logging.warning("无效的 SQL 查询，跳过执行。")
        return pd.DataFrame()

    conn = sqlite3.connect(db_file)
    try:
        # 使用参数化查询，防止 SQL 注入攻击
        result_df = pd.read_sql_query(sql_query, conn)
        result_str = result_df.to_string(index=False)
        logging.info(f"SQL 执行成功，查询结果：{result_str}")
    except sqlite3.Error as e:
        logging.error(f"SQL 执行出错: {e}")
        result_df = pd.DataFrame()
        result_str = str(e)
    finally:
        conn.close()

    context.append({
        "question": question,
        "sql": sql_query,
        "result": result_str
    })
    # 将查询结果发送给 GPT
    send_results_to_gpt(question, sql_query, result_str)
    return result_df

# 主程序
def main():
    # 建立資料索引、大綱並轉換 CSV
    data_index, data_outline = build_data_index_and_convert(DOCS_FOLDER, DB_FILE, force_update=False)

    if not data_index or not data_outline:
        print("未生成资料索引或大纲，请检查文件。")
        return
    print(f"生成的大纲:\n{data_outline}")  # 在 main 中确认输出大纲
    while True:
        question = input("请输入您的问题（输入 '退出' 结束）：")
        if question.strip().lower() == '退出':
            # 在用户结束会话时清空 context
            context.clear()
            print("会话已结束，上下文已清空。")
            break

        # 生成 SQL 查询语句
        sql_query, error_message = generate_sql_query(data_outline, question, context)
        if error_message:
            print(f"无法生成 SQL 查询: {error_message}")
            continue  # 跳过后续执行
        
        print(f"生成的 SQL 查询: {sql_query}")

        # 执行 SQL 查询并保存上下文
        result = execute_sql_query(DB_FILE, sql_query, question, context)
        if not result.empty:
            print(f"查询结果:\n{result}")
        else:
            print("未找到匹配的数据。")    

if __name__ == "__main__":
    main()