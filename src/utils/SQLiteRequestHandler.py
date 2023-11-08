import sqlite3, socketserver, pickle, os, time, sys, socket, errno, re
sys.path.append("./src/")
from utils.utility import *

# # Check if cruz in pwd
# if "scratch" in os.getcwd():
#     DB_PATH = "./results/main_daint.db"
# elif "cruz" in os.getcwd() and not "scratch" in os.get_cwd():
#     DB_PATH = "./results/main_sam_local.db"
# else:
#     DB_PATH = "./results/TEMP.db"

class SQLiteRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        data_array = []
        self.request.setblocking(0)       
        while True:
            try:
                packet = self.request.recv(4096)
                packet = str(packet)
                packet = packet[2:-1]
                data_array.append(packet)
                if 'END' in packet:
                    match = re.search(r'START(.*?)END', ''.join(data_array))
                    data_str = match.group(1)
                    data_array = bytes(data_str, "utf-8").decode("unicode_escape").encode("latin-1")
                    break

            except Exception as e:
                print(e)

        request_data = pickle.loads(data_array)
        sql_query = request_data['sql_query']
        parameters = tuple(request_data.get('parameters', []))

        while True:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            try:
                cursor.execute(sql_query, parameters)
                if sql_query.lower().startswith("select")  or sql_query.lower().startswith("pragma"):
                    result = cursor.fetchall()
                else:
                    conn.commit()
                    result = ['worked']
            except sqlite3.Error as e:
                result = str(e)
            
            conn.close()
            if result != "database is locked":
                self.request.sendall(b'START'+pickle.dumps(result)+b'END')
                break
            else:
                time.sleep(0.1)
                print('database is busy, waiting...')



if __name__ == "__main__":
    args = parse_args()
    DB_PATH = args.db_path

    # _____ SAME FIELDS AS IN Power_Dataframe.py _____
    key_fields = ['dataset','optimizer_name','optimizer_params','mb_size','pretraining_status','network_class_str','network_input_params','overlap_ratio','loss_params','loss_class_str','seed']
    key_fields_type = ['TEXT',      'TEXT',         'TEXT',       'INTEGER',     'REAL',            'TEXT',                  'TEXT',              'REAL',       'TEXT',    'TEXT',    'INTEGER']
    non_key_fields = ['results','net_state', 'optimizer_fun','opt_class_str','loss_fun'] 
    non_key_fields_type = ['BLOB',  'BLOB',    'BLOB',          'BLOB',        'BLOB']
    pretraining_lvl = [70, 90]
    pretraining_lvl_fields = ['dataset','seed','net_state']
    pretraining_lvl_fields_type = ['TEXT','INTEGER','BLOB']
    # _____ SAME FIELDS AS IN Power_Dataframe.py _____
    
    def __create_tables():
        DB = sqlite3.connect(DB_PATH)
        fields = key_fields + non_key_fields
        fields_type = key_fields_type + non_key_fields_type
        c = DB.cursor()
        query = f"CREATE TABLE IF NOT EXISTS results ({', '.join(f'{field} {field_type}' for field,field_type in zip(fields,fields_type))})"
        c.execute(query)
        for lvl in pretraining_lvl:
            query = f"CREATE TABLE IF NOT EXISTS net_state_{lvl} ({', '.join(f'{field} {field_type}' for field,field_type in zip(pretraining_lvl_fields, pretraining_lvl_fields_type))})"
            c.execute(query)
        DB.commit()
        DB.close()
        
    __create_tables()

    HOST, PORT = "localhost", 9999
    # server = socketserver.TCPServer((HOST, PORT), SQLiteRequestHandler(DB_PATH))
    server = socketserver.TCPServer((HOST, PORT), SQLiteRequestHandler)
    print("Server ready to serve...")
    server.serve_forever()
    print("Server says: au revoir...")



# INSERT INTO results 
# (dataset, optimizer_name, optimizer_params, mb_size, pretraining_status, network_class_str, network_input_params, overlap_ratio, loss_params, loss_class_str, seed, results, net_state, optimizer_fun, opt_class_str, loss_fun) 
# VALUES 
# ('a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a')