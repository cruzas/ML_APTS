import torch
import pandas as pd
import numpy as np
import os,sys,re,itertools,pickle,ast,time,torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
sys.path.append("./src/")
from utils.utility import *
from optimizers.TR import *
from data_loaders.Power_DL import *
from data_loaders.OverlappingDistributedSampler import *
from optimizers.APTS import *   
from optimizers.APTS_D import *
from custom_datasets.Sine import *
import sqlite3,socket

from models.simple_nn import *
# base_path = os.getcwd()
# sys.path.append(os.path.abspath(base_path+'\\optimizers'))
# from ..optimizers.TR import *
# from utility import *






class Power_Dataframe():
    def __init__(self, results_filename='./results/SQLite_Tests_DB.db', data_dir=os.path.abspath("./data"), sequential=True, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), save_optimizer_and_net_status=False, save_pretrained_net_state=False, 
                 DEBUG=False, enable_hints=True, comments=True, regression=False, HOST=None, PORT=None):
        '''
        Note: "__x" functions: are for internal use only
        '''
        self.DEBUG = DEBUG
        if DEBUG:
            self.results_filename = '_debug.db'.join(results_filename.split('.db'))
        else:
            self.results_filename = results_filename
            
        self.HOST = HOST
        self.PORT = PORT
        
        self.enable_hints = enable_hints
        self.save_optimizer_and_net_status = save_optimizer_and_net_status
        self.save_pretrained_net_state = save_pretrained_net_state
        self.device = device
        self.regression = regression
        
        self.key_fields = ['dataset','optimizer_name','optimizer_params','mb_size','pretraining_status','network_class_str','network_input_params','overlap_ratio','loss_params','loss_class_str','seed']
        self.key_fields_type = ['TEXT',      'TEXT',         'TEXT',       'INTEGER',     'REAL',            'TEXT',                  'TEXT',              'REAL',       'TEXT',    'TEXT',    'INTEGER']
        # -> if "class" is in the name it gets encoded to avoit problems with e.g. \n and search process messes up 

        self.non_key_fields = ['results','net_state', 'optimizer_fun','opt_class_str','loss_fun'] 
        self.non_key_fields_type = ['BLOB',  'BLOB',    'BLOB',          'BLOB',        'BLOB']
        if not regression:
            self.pretraining_lvl = [70, 90]
        else:
            self.pretraining_lvl = [3, 2] # these are later divided by 10 so that it corresponds to 0.3 and 0.2
        self.pretraining_lvl_fields = ['dataset','seed','net_state']
        self.pretraining_lvl_fields_type = ['TEXT','INTEGER','BLOB']
        self.seed_interval = 1000  # Distance between tested seeds in trials
        self.sequential = sequential
        self.comments = comments
        self.data_dir = data_dir


        def _key_query_maker(key_dict,seed=True):
            values = ''
            for col in self.key_fields:
                if key_dict.get(col) is not None and key_dict.get(col) != '':
                    if seed is False and col == 'seed':
                        pass
                    else:
                        values += f'{col}=? AND '
            return values[:-5]
        self.key_query_maker = lambda key_dict,seed: _key_query_maker(key_dict,seed)

        def _data_manipulation(key_dict):
            values = []
            for col in self.key_fields:
                if key_dict.get(col) is not None and key_dict.get(col) != '':
                    if type(key_dict.get(col)) is dict:
                        values.append(self.__dict_to_str(key_dict[col]))
                    elif 'class' in col:
                        values.append(key_dict[col].encode('unicode_escape').decode())
                    else:
                        values.append(key_dict[col])
            return values
        self.data_manipulation = lambda key_dict: _data_manipulation(key_dict)

        self.__create_tables()


    def __query_db(self, sql_query, parameters=[]):
        for i in range(len(parameters)):
            if type(parameters[i]) not in [str, int, float, bytes]:
                parameters[i] = pickle.dumps(parameters[i])
        if parameters == []:
            request_data = {
                "sql_query": sql_query,
                "parameters": ''
            }
        else:
            request_data = {
                "sql_query": sql_query,
                "parameters": parameters
            }
        if self.HOST is not None and self.PORT is not None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.HOST, self.PORT))
                sock.sendall(b'START'+pickle.dumps(request_data)+b'END')
                data_array = []
                sock.setblocking(0)      
                while True:
                    try:
                        packet = sock.recv(4096)
                        packet = str(packet)
                        packet = packet[2:-1]
                        data_array.append(packet)
                        if 'END' in packet:
                            match = re.search(r'START(.*?)END', ''.join(data_array))
                            data_str = match.group(1)
                            r = bytes(data_str, "utf-8").decode("unicode_escape").encode("latin-1")
                            break
                    except:
                        pass

            if r != b'':
                result = pickle.loads(r)
                for i in range(len(result)):
                    result[i] = list(result[i])
                    for j in range(len(result[i])):
                        if type(result[i][j]) is bytes:
                            result[i][j] = pickle.loads(result[i][j])
            else:
                result = []
        else:
            conn = sqlite3.connect(self.results_filename)
            cursor = conn.cursor()
            cursor.execute(sql_query, parameters)
            if sql_query.lower().startswith("select") or sql_query.lower().startswith("pragma"):
                result = cursor.fetchall()
            else:
                conn.commit()
                result = []          
            conn.close()

            if result != []:
                for i in range(len(result)):
                    result[i] = list(result[i])
                    for j in range(len(result[i])):
                        if type(result[i][j]) is bytes:
                            result[i][j] = pickle.loads(result[i][j])
            
        return result


    def merge_databases(self, DB2):
        try:
            # Get a list of table names in the source database
            source_tables = DB2.__query_db("SELECT name FROM sqlite_master WHERE type='table';")
            # Loop through each table and copy its content to the destination database
            for table in source_tables:
                table_name = table[0]
                rows = DB2.__query_db(f"SELECT * FROM {table_name};")

                # Get the column names for the table
                column_names = DB2.__query_db(f"PRAGMA table_info({table_name});")
                columns = [column[1] for column in column_names]

                # Create the corresponding table in the destination database using __query_db
                query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)});"
                self.__query_db(query, parameters=[])

                # Insert the rows into the destination table using __query_db
                for row in rows:
                    #first check if the row is already in the DB through __get_row_id
                    rowid = self.__get_row_id({col:row[i] for i,col in enumerate(columns)}, table_name)
                    print(rowid)
                    if rowid == []:
                        insert_query = f"INSERT INTO {table_name} VALUES ({', '.join(['?'] * len(columns))});"
                        self.__query_db(insert_query, parameters=row)

            print("Database merge completed successfully.")

        except Exception as e:
            print(f"An error occurred during database merge: {str(e)}")


    def __get_row_id(self, key_dict, table):
        query = f"SELECT rowid FROM {table} WHERE "+self.key_query_maker(key_dict,seed=True if 'seed' in key_dict.keys() else False)
        parameters = self.data_manipulation(key_dict)
        results = self.__query_db(query, parameters)
        if len(results) == 0:
            return []
        else:
            return [r[0] for r in results]
        
    def __save_in_DB(self, key_dict, table, data_field, data):
        if data_field not in self.non_key_fields:
            raise ValueError(f'Error: {data_field} is not a valid field')
        if type(data) is dict:
            data = self.__dict_to_str(data) 
        results = self.__get_row_id(key_dict, table)
        if len(results) == 0:
            query = f"INSERT INTO {table} ({', '.join(self.key_fields)},{data_field}) VALUES (?,{', '.join('?' for i in range(len(self.key_fields)))})"
            parameters = self.data_manipulation(key_dict)
            results = self.__query_db(query, parameters+[data])
        elif len(results) == 1:
            query = f"UPDATE {table} SET {data_field}=? WHERE rowid = {results[0]}"
            results = self.__query_db(query, [data])
        else:
            raise ValueError(f'Error: {len(results)} rows found for key {key_dict}. Only one is allowed.')
        
        # results = self.__get_row_id(key_dict, table)
        # assert len(results) == 1, f'Error: {len(results)} rows found for key {key_dict}'
        # query = f"UPDATE {table} SET {data_field}=? WHERE rowid = {results[0]}"
        # results = self.__query_db(query, [data])


    # def __load_from_DB(self, key_dict, table, data_field):
    #     if data_field not in self.non_key_fields:
    #         raise ValueError(f'Error: {data_field} is not a valid field')
    #     c=self.DB.cursor()
    #     results = self.__get_row_id(key_dict, table)
    #     assert len(results) <= 1, f'Error: {len(results)} rows found for key {key_dict}. Only one is allowed.'
    #     if len(results) == 0:
    #         return []
    #     else:
    #         query = f"SELECT {data_field} FROM {table} WHERE rowid = {results[0]}"
    #         c.execute(query)
    #         results = c.fetchall()
    #         c.close()
    #         return pickle.loads(results[0])
        
    def __dict_to_str(self, dictt):
        dictt = self.__sort_dictionary(dictt)
        
        for key in dictt.keys(): # here we look for class in dictt
            if 'class' in str(dictt[key]):
                # dictt[key] = pickle.dumps(dictt[key]) #TODO: save the serialized file into a different field of DB
                dictt[key] = f'##{dictt[key].__name__}'
        dict_str = str(dictt).replace(' ','').replace("inf", '2e308') # The only workaround I found to save inf such that it can be read through ast
        #check if this dictionary can be red through ast:
        try:
            if ast.literal_eval(dict_str) == dictt:
                return dict_str
            else:
                dictt2 = ast.literal_eval(dict_str)
                for key in dictt.keys():
                    if dictt[key] != dictt2[key] and dictt2[key]=='inf':
                        dictt2[key] = float('inf')
                    elif dictt2[key] is not None:
                        raise ValueError(f'Error:\n{dictt[key]}\n!=\n{dictt2[key]}')
                return dict_str
        except Exception as e:
            raise ValueError(f'Error:\n{e}')
        

    def __str_to_dict(self, dict_str):
        dictt = ast.literal_eval(dict_str)
        return dictt

    #____________________________________________________________________________________________________________________________________________________________________
    # NOTE: THIS PART CAN BE MODIFIED TO ADD NEW DATAFRAMES.. REMEMBER TO ADD THE TOTAL AMOUNT OF SAMPLES
    #____________________________________________________________________________________________________________________________________________________________________
    def __load_data(self, dataset="mnist", data_dir=os.path.abspath("./data"), TOT_SAMPLES=False):
        '''
        Loads the dataset. Modify this code if you want to add more datasets.
        '''
        if dataset.lower() == 'mnist':
            if TOT_SAMPLES is False:
                train_set = datasets.MNIST(root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
                test_set = datasets.MNIST(root=data_dir, train=False, transform=transforms.ToTensor(), download=True)
            else:
                return 60000
        elif dataset.lower() == 'cifar10':
            if TOT_SAMPLES is False:
                train_set = datasets.CIFAR10(root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
                test_set = datasets.CIFAR10(root=data_dir, train=False, transform=transforms.ToTensor(), download=True)
            else:
                return 50000
        elif dataset.lower() == 'sine':
            num_train_samples = int(1000)
            if TOT_SAMPLES is False:
                num_test_samples = int(0.5 * 1000)
                input_vector_length = 50
                train_set = Sine(num_train_samples, input_vector_length)
                test_set = Sine(num_test_samples, input_vector_length)
            else:
                return num_train_samples
        else:
            raise ValueError('Unsupported dataset. Currently supported datasets are: MNIST, CIFAR10')

        return train_set, test_set
    #____________________________________________________________________________________________________________________________________________________________________

    def del_unallocated_memory(self):
        self.__query_db("VACUUM", parameters=[])

    def weight_of_a_row(self,row_index=None):
        if row_index is None:
            return sys.getsizeof(self.__query_db(f"SELECT * FROM results", parameters=[]))
        else:
            return sys.getsizeof(self.__query_db(f"SELECT * FROM results WHERE rowid = {row_index}", parameters=[]))

    def __create_tables(self):
        fields = self.key_fields + self.non_key_fields
        fields_type = self.key_fields_type + self.non_key_fields_type
        query = f"CREATE TABLE IF NOT EXISTS results ({', '.join(f'{field} {field_type}' for field,field_type in zip(fields,fields_type))})"
        self.__query_db(query, parameters=[])
        for lvl in self.pretraining_lvl:
            query = f"CREATE TABLE IF NOT EXISTS net_state_{lvl} ({', '.join(f'{field} {field_type}' for field,field_type in zip(self.pretraining_lvl_fields, self.pretraining_lvl_fields_type))})"
            self.__query_db(query, parameters=[])
    
    def __del_opt__(self, optimizer_name):
        rowids,seeds = self.__select_results(key=None, dataset=None, optimizer_name=optimizer_name, optimizer_params=None, mb_size=None, pretraining_status=None, network_class_str=None, overlap_ratio=None)
        # delete every row_id in rowid from DB:
        # c = self.DB.cursor()
        for r in rowids:
            # c.execute(f"DELETE FROM results WHERE rowid = {r[0]}")
            self.__query_db(f"DELETE FROM results WHERE rowid = {r[0]}", parameters=[])
        # self.DB.commit();c.close()
        print(f'Optimizer {optimizer_name} has been erased from Power_Dataframe')
        return

    def __del_db__(self):
        # self.DB.close()
        os.remove(self.results_filename)
        self.__create_tables()
        print('DB has been erased')


    # Get train and test loaders
    def __create_dataloaders(self, dataset="mnist", data_dir=os.path.abspath("./data"), mb_size=100, overlap_ratio=0):
        train_set, test_set = self.__load_data(dataset = dataset, data_dir=data_dir)
        mb_size = min(len(train_set), int(mb_size))
        mb_size2 = min(len(test_set), int(mb_size))

        if self.sequential:
            train_loader = Power_DL(dataset=train_set, shuffle=True, device=self.device, minibatch_size=mb_size, overlapping_samples=overlap_ratio)
            test_loader = Power_DL(dataset=test_set, shuffle=False, device=self.device, minibatch_size=mb_size2)
        else:
            world_size = world_size = dist.get_world_size() if dist.is_initialized() else 1
            overlapping_samples = int(overlap_ratio * mb_size)
            local_batch_size = int(mb_size/world_size + overlapping_samples*(world_size - 1)/2)
            local_test_batch_size = int(mb_size/world_size + overlapping_samples*(world_size - 1)/2)

            train_sampler = OverlappingDistributedSampler(train_set, num_replicas=world_size, shuffle=True, overlapping_samples=overlapping_samples)
            test_sampler = OverlappingDistributedSampler(test_set, num_replicas=world_size, shuffle=False, overlapping_samples=overlapping_samples)

            train_loader = DataLoader(train_set, batch_size=local_batch_size, sampler=train_sampler, drop_last=False)
            test_loader = DataLoader(test_set, batch_size=local_test_batch_size, sampler=test_sampler, drop_last=False)
        
        return train_loader, test_loader


    def __compute_accuracy(self, data_loader, net):
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Reduce accuracy across all processes
        if dist.is_initialized() and dist.get_world_size() > 1:
            correct = torch.tensor(correct, device=self.device)
            total = torch.tensor(total, device=self.device)
            dist.all_reduce(correct)
            dist.all_reduce(total)
            correct = correct.cpu().item()
            total = total.cpu().item()

        accuracy = 100 * correct / total
        return accuracy


    def __compute_loss(self, data_loader, net, criterion):
        with torch.no_grad():
            loss = 0
            count = 0
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = net(inputs)  # Forward pass
                loss += criterion(outputs, labels)  # Compute the loss
                count += 1

            # Reduce loss across all processes
            if dist.is_initialized() and dist.get_world_size() > 1:
                count = torch.tensor(count, device=self.device)
                dist.all_reduce(loss)
                dist.all_reduce(count)
                # Not doing loss=loss.item() since it's done afterwards in line 171
                count = count.cpu().item()

            return loss.cpu().item() / count


    def __do_one_optimizer_test(self, train_loader, test_loader, optimizer, net=None, num_epochs=50, criterion=torch.nn.CrossEntropyLoss(), desired_accuracy=100, counter=[], key_dict={}, epoch_start=0, seed=None):
        df_data = {'accuracy': [], 'loss': [], 'time': []}  # Data to be saved in dataframe

        # Save network state when pretraining_status stage accuracy reached
        # Starting from 0 so that we can measure the starting loss
        if epoch_start > 0:
            epoch_start += 1
        for epoch in range(epoch_start, num_epochs + 1):
            epoch_train_time = 0
            epoch_train_loss = 0
            count = 0
            # Compute epoch train loss and train time, and test accuracy
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if "TR" in optimizer.__class__.__name__:
                    def closure():
                        optimizer.zero_grad()
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        if torch.is_grad_enabled():
                            loss.backward()
                        if dist.is_initialized() and dist.get_world_size() > 1:
                            # Global loss
                            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                            loss /= dist.get_world_size()
                            average_gradients(net)
                        return loss  
                else:
                    def closure():
                        # Zero the parameter gradients
                        optimizer.zero_grad()
                        # Forward + backward + optimize
                        outputs = net(inputs)
                        train_loss = criterion(outputs, labels)
                        if torch.is_grad_enabled():
                            train_loss.backward()
                            if dist.is_initialized() and dist.get_world_size() > 1:
                                # TODO: add all losses together?
                                # Average gradients in case we are using more than one CUDA device
                                average_gradients(net) 
                        return train_loss              

                # Add to total training time
                start = time.time()
                if epoch == 0:
                    with torch.no_grad():
                        train_loss = self.__compute_loss(data_loader=train_loader, net=net, criterion=criterion)
                else:
                    if 'APTS' in str(optimizer) and 'W' in str(optimizer):
                        train_loss = optimizer.step(inputs, labels)[0]
                    elif 'APTS' in str(optimizer) and 'D' in str(optimizer):
                        train_loss = optimizer.step(inputs, labels)
                    else:
                        train_loss = optimizer.step(closure)
                        if type(train_loss) is list or type(train_loss) is tuple:
                            train_loss = train_loss[0]
                    if 'torch' in str(type(train_loss)):
                        train_loss = train_loss.cpu().detach().item()
                end = time.time()

                # NOTE: We might want to change this if we also want to take into account the time needed to compute the starting loss and accuracy
                epoch_train_time += (0 if epoch == 0 else (end - start)/1000)
                epoch_train_loss += train_loss
                count += 1
    
            # epoch_train_loss, epoch_train_time = self.do_one_optimizer_train_epoch(epoch, train_loader, optimizer, net, criterion)
            if 'MNIST' in str(test_loader.dataset) or 'CIFAR' in str(test_loader.dataset):
                test_accuracy = self.__compute_accuracy(data_loader=test_loader, net=net)
            else: # Regression
                test_accuracy = self.__compute_loss(data_loader=test_loader, net=net, criterion=criterion)

            df_data['accuracy'].append(test_accuracy)
            if isinstance(epoch_train_loss, torch.Tensor):
                epoch_train_loss = epoch_train_loss.cpu()
                df_data['loss'].append(epoch_train_loss.detach().item()/count)
            else:
                df_data['loss'].append(epoch_train_loss/count)

            df_data['time'].append(epoch_train_time)

            if 'MNIST' in str(test_loader.dataset) or 'CIFAR' in str(test_loader.dataset):
                if self.save_optimizer_and_net_status:
                    for i, lvl in enumerate(self.pretraining_lvl):
                        if test_accuracy >= lvl and self.save_pretrained_net_state:
                            counter[i] += 1
                            net_state_key = self.__dict_to_keyNET(key_dict)
                            if (net_state_key not in self.df['net_state_'+str(lvl)].keys()):
                                self.df[f'net_state_{lvl}'][net_state_key] = {}
                                self.df[f'net_state_{lvl}'][net_state_key][str(seed)] = net.state_dict()
                            elif counter == 1:
                                pass
                                #TODO: key already exists and this is the first time we go through this step, 
                                #maybe check if the NN has different parameters than the ones already saved and save this too (on a different seed)             

                # Check if test accuracy meets desired accuracy. If so, break training loop.
                if test_accuracy >= desired_accuracy:
                    break

            else: # Regression
                if self.save_optimizer_and_net_status:
                    for i, lvl in enumerate(self.pretraining_lvl):
                        if test_accuracy <= lvl/10 and self.save_pretrained_net_state:
                            counter[i] += 1
                            net_state_key = self.__dict_to_keyNET(key_dict)
                            if (net_state_key not in self.df['net_state_'+str(lvl)].keys()):
                                self.df[f'net_state_{lvl}'][net_state_key] = {}
                                self.df[f'net_state_{lvl}'][net_state_key][str(seed)] = net.state_dict()
                            elif counter == 1:
                                pass
                                #TODO: key already exists and this is the first time we go through this step, 
                                #maybe c                

        # Pandas dataframe containing all necessary data: epoch, test accuracy, training loss, and training time
        return df_data           
    

    def __retrieve_data_from_DB(self, rowids):
        # c = self.DB.cursor()
        query = f"SELECT {','.join(self.key_fields)} FROM results WHERE rowid IN ({','.join(str(r) for r in rowids)})"
        # c.execute(query)
        # results = c.fetchall()
        # c.close()
        results = self.__query_db(query, parameters=[])
        return results

    def __get_field_from_rowid(self, rowid, field):
        # c = self.DB.cursor()
        query = f"SELECT {field} FROM results WHERE rowid = {rowid}"
        # c.execute(query)
        # results = c.fetchall()
        # c.close()
        results = self.__query_db(query, parameters=[])
        return results[0][0]

    def __merge_seeds(self, values, epochs, mode='optimizer_params'):
        '''
        The only difference in different elements of values are the optimizer_params and seeds. This function collects all the results with same optimizer_params and merges them into one.
        values = [[rowid,results,net_state,optimizer_fun,loss_fun],...]
        '''
        data = self.__retrieve_data_from_DB([v[0] for v in values])
        results = []; indexes = list(range(len(data))); c=0; IIndexes = indexes.copy()
        for i in IIndexes:
            if i in indexes:
                indexes.pop(indexes.index(i))
                if len(values[i][1]['accuracy'])>=epochs+1:
                    results.append({'accuracy':values[i][1]['accuracy'].head(epochs+1), 'loss':values[i][1]['loss'].head(epochs+1), 'time':values[i][1]['time'].head(epochs+1)}) # pandas dataframe with accuracy, loss and time
                    results[-1][mode] = self.__get_field_from_rowid(values[i][0], mode)
                    results[-1]['optimizer_name'] = self.__get_field_from_rowid(values[i][0], 'optimizer_name')
                    Indexes=indexes.copy()
                    for j in Indexes:
                        if j in indexes and all([a==b for ii,(a,b) in enumerate(zip(data[i],data[j])) if ii==self.key_fields.index(mode)]):
                            indexes.pop(indexes.index(j))
                            if len(values[j][1]['accuracy'])>=epochs+1:
                                results[-1]['accuracy'] = pd.concat([results[-1]['accuracy'],values[j][1]['accuracy'].head(epochs+1)], axis=1)
                                results[-1]['loss'] = pd.concat([results[-1]['loss'],values[j][1]['loss'].head(epochs+1)], axis=1)
                                results[-1]['time'] = pd.concat([results[-1]['time'],values[j][1]['time'].head(epochs+1)], axis=1)
        return results
                            
        
    def __find_optimal_params(self, epochs, mode='median', measure='accuracy', IGNORED_FIELDS=None, **kwargs):
        values,_ = self.__select_results(IGNORED_FIELDS,**kwargs)
        if len(values) == 0:
            raise ValueError('Error: no trials found for these Optimizer')
        results = self.__merge_seeds(values,epochs, mode='optimizer_params')
        trials_per_key = [len(r['accuracy'].columns) for r in results]
        if min(trials_per_key) < 10:
            print(f'Optimal params may be inaccurate due to low amount of trials (min amount = {min(trials_per_key)})')

        df2 = []; score = []
        for res in results:
            if 'median' in mode:
                df2.append(res[measure].median(axis=1))
            elif '-mean' in mode:
                df2.append(res[measure].apply(lambda row: np.mean(row[row >= np.percentile(row, float(mode.split('-')[0]) * 100)]), axis=1))
            elif '_mean' in mode:
                df2.append(res[measure].apply(lambda row: np.mean(row[row >= np.percentile(row, float(mode.split('_')[0]) * 100)]), axis=1))
            else:
                df2.append(res[measure].mean(axis=1))
            
            if measure == "accuracy":
                score.append(sum(df2[-1]))
            else:
                score.append(-sum(df2[-1]))

            
        print(f"Optimal parameters for {results[score.index(max(score))]['optimizer_name']} found: {results[score.index(max(score))]['optimizer_params']}")
        return ast.literal_eval(results[score.index(max(score))]['optimizer_params'])


    def __select_results(self, IGNORED_FIELDS=None, **kwargs):
        '''
        Looks for all keys in DF with same values except for the non-specified ones. 
        E.g. if you don't specify the seed than this function retrieves every test with same dataset,optimizer,...
        with different starting seed, i.e., it retrieves each trial.
        TODO: Improve explanation
        '''
        dictt = {}; counter = 0
        for key in self.key_fields:
            if kwargs.get(key) is not None:
                counter += 1
                dictt[key] = kwargs[key]
            else:
                dictt[key] = None

        assert len(dictt) > 0, 'Error: no key specified'

        # c = self.DB.cursor()       
        if IGNORED_FIELDS is not None: 
            for item in IGNORED_FIELDS:
                dictt[item] = ''
        query = f"SELECT rowid,seed,{','.join(self.non_key_fields)} FROM results WHERE "+self.key_query_maker(dictt,seed=False)
        parameters = self.data_manipulation(dictt)
        # c.execute(query,parameters)
        results = self.__query_db(query, parameters)
        # results = c.fetchall()
        if len(results) == 0:
            return [],[]
        else:
            values=[];seeds=[];epochs=[]
            for r in results:
                values.append([r[0]] + [r[i]  if r[i] is not None else None for i in range(2,len(r))])
                seeds.append(int(r[1]))
                epochs.append(len(values[-1][1+self.non_key_fields.index('results')]['accuracy'])-1) # Removing 1 epoch because it the 0th epoch is before the training starts

            if counter == len(self.key_fields)-1: #ignoring "seed"
                values = [x for _,x in sorted(zip(epochs,values),reverse=True)] #sort results by amount of epochs in descending order
            return values,seeds
    

    def __get_net_state(self, key_dict, seed):
        '''
        Returns the net state for a given key and seed
        '''
        key_dict2 = key_dict.copy()
        key_dict2['seed'] = seed
        query = f"SELECT net_state FROM results WHERE "+self.key_query_maker(key_dict2,seed=True)
        results = self.__query_db(query, parameters=[])
        return results[0][0]
    

    def __do_tests(self, df, key_dict, epochs, criterion=None):
        counter = [0]*len(self.pretraining_lvl)
        for seed in df.keys(): # Loops over different seeds (different trials)
            seed = int(seed); torch.manual_seed(seed)
            key_dict['seed'] = seed
            opt_fun = key_dict['opt_fun']

            #--# Check if some epochs are already computed
            if len(df[str(seed)])-1 == epochs: # Skip this test
                print('This should never happen')
                continue 
            elif len(df[str(seed)])-1 > epochs:
                raise ValueError('Error: This shouldnt happen.')
            
            train_loader, test_loader = self.__create_dataloaders(dataset=key_dict['dataset'], data_dir=self.data_dir, mb_size=key_dict['mb_size'], overlap_ratio=key_dict['overlap_ratio'])
            net = key_dict['network_function'](**key_dict['network_input_params'])
            net.to(self.device)
            if len(df[str(seed)]) > 0: # Tests are already initiated, proceed by loading net state and optimizer state
                pass
                # TODO: still a todo
                # ep_done = len(df[str(seed)])-1 
                # accuracy = list(df[str(seed)]['accuracy'])
                # loss = list(df[str(seed)]['loss'])
                # t = list(df[str(seed)]['time'])

                # net.load_state_dict(self.__get_net_state(key_dict,str(seed)))
                # opt = self.df['optimizer_fun'][key_str][str(seed)]
                # # opt.param_groups[0]['params'] = net.parameters()
                # opt.add_param_group({'params': net.parameters()})
            else: # Start from scratch
                ep_done = 0
                accuracy = []; loss = []; t = []
                #____________________________________________________________________________________________________________________________________________________________________
                # NOTE: MODIFY THIS PART TO ALLOW DIFFERENT INPUTS FOR YOUR OPTIMIZERS.
                #____________________________________________________________________________________________________________________________________________________________________
                try:
                    params = copy.deepcopy(key_dict['optimizer_params'])
                    if 'APTS' not in key_dict['optimizer_name']:
                        opt = opt_fun(net.parameters(), **params)
                    elif 'APTS' in opt_fun.__name__ and 'W' in opt_fun.__name__:
                        opt = opt_fun(net.parameters(), model=net, loss_fn=criterion, device=self.device, **params)
                    elif 'APTS' in opt_fun.__name__ and 'D' in opt_fun.__name__:
                        opt = opt_fun(net.parameters(), model=net, loss_fn=criterion, device=self.device, **params)
                    else:
                        opt = opt_fun(net.parameters(), model=net, **params)
                #____________________________________________________________________________________________________________________________________________________________________
                except Exception as e:
                    if 'got multiple values for argument' in str(e):
                        raise ValueError(f'Error: {opt_fun.__name__} has multiple values for argument {str(e).split("argument ")[1].split(" ")[0]}. You probably forgot to add {str(e).split("argument ")[1].split(" ")[0]} to the "ignore_optimizer_params" list.')
                    else:
                        raise ValueError(f'Error: {opt_fun.__name__} raised an exception: {e}')

            #--# Start training
            if 'MNIST' in str(train_loader.dataset) or 'CIFAR10' in str(train_loader.dataset):
                desired_accuracy = 100
            else: # Regression
                desired_accuracy = 1e-3

            data = self.__do_one_optimizer_test(train_loader, test_loader, opt, net=net, num_epochs=epochs, criterion=criterion, desired_accuracy=desired_accuracy, 
                                              counter=counter, key_dict=key_dict, epoch_start=ep_done, seed=seed)

            accuracy.extend(data['accuracy'])
            loss.extend(data['loss'])
            t.extend(data['time'])
            data['accuracy'] = accuracy
            data['loss'] = loss
            data['time'] = t

            #--# Saving results
            if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
                self.__save_in_DB(key_dict, table='results', data_field='results', data=pd.DataFrame(data))
                self.__save_in_DB(key_dict, table='results', data_field='loss_fun', data=criterion)
                if self.save_optimizer_and_net_status:
                    self.__save_in_DB(key_dict, table='results', data_field='net_state', data=net.state_dict())
                    self.__save_in_DB(key_dict, table='results', data_field='optimizer_fun', data=opt)

            df[str(seed)] = pd.DataFrame(data)
        return df
    
    def __input_params_check(self,dataset=None, mb_size=None, opt_fun=None, optimizer_params=None, pretraining_status=None, overlap_ratio=None, network_fun=None, network_params=None, loss_fun=None, 
                             loss_params=None, ignore_optimizer_params=None, **kwargs):              
        mb_size = mb_size if type(mb_size) is list else [mb_size]
        opt_fun = opt_fun if type(opt_fun) is list else [opt_fun]
        optimizer_name = [opt_name.__name__ for opt_name in opt_fun]
        optimizer_params = optimizer_params if type(optimizer_params) is list else [optimizer_params]
        network_fun = network_fun if type(network_fun) is list else [network_fun]
        network_params = network_params if type(network_params) is list else [network_params]
        overlap_ratio = overlap_ratio if type(overlap_ratio) is list else [overlap_ratio]

        Tot_samples = self.__load_data(dataset=dataset, data_dir=self.data_dir, TOT_SAMPLES=True)
        for i in range(len(mb_size)):
            if str(mb_size[i]).isalpha():
                mb_size[i].lower()
                assert mb_size[i] == 'full', "Error: mb_size must be an integer or 'full'."
            if mb_size[i] == 'full' or mb_size[i] >= Tot_samples:
                mb_size[i] = Tot_samples
                if overlap_ratio[i] != 0:
                    overlap_ratio[i] = 0

        if type(ignore_optimizer_params[0]) is not list:
            ignore_optimizer_params=[ignore_optimizer_params for _ in range(len(opt_fun))]
        assert len(opt_fun) == len(ignore_optimizer_params), "Error: length of optimizer list is not equal to length of ignore_optimizer_params list."

        if len(network_fun) != len(network_params) and len(network_fun)!=1 and len(network_params)!=1:
            raise ValueError('Error: length of network_fun list is not equal to length of network_params list.\nYou can have:\n- 1 network_fun and many network_params;\n- many network_fun and 1 network_params\n- x network_fun and x network_params (each network_fun with its own network_params).')
        if len(opt_fun) != len(optimizer_params) and len(opt_fun)!=1 and len(optimizer_params)!=1:
            raise ValueError('Error: length of opt_fun list is not equal to length of optimizer parameters list.\nYou can have:\n- 1 opt_fun and many optimizer_params;\n- many opt_fun and 1 optimizer_params\n- x optimizer and x optimizer_params (each opt_fun with its own optimizer_params).')
        
        if len(opt_fun)==1 and len(optimizer_params)!=1:
            opt_fun=opt_fun*len(optimizer_params)
            optimizer_name=optimizer_name*len(optimizer_params)
            ignore_optimizer_params=ignore_optimizer_params*len(optimizer_params)
        elif len(optimizer_params)==1 and len(opt_fun)!=1:
            optimizer_params=optimizer_params*len(opt_fun)

        if len(network_fun)==1 and len(network_params)!=1:
            network_fun=network_fun*len(network_params)
        elif len(network_params)==1 and len(network_fun)!=1:
            network_params=network_params*len(network_fun)

        network_class_str = []; total_network_params = []
        for i,nn in enumerate(network_fun):
            try:
                network_class_str.append( inspect.getsource(nn) )
            except:
                raise ValueError('Error: network_fun must be a function class not the defined neural network with input parameters (e.g. it should be just "NN", and not "NN(params)" ).')
            total_network_params.append(self.__initial_class_params_retriever(nn))
            #check if network_params contains a key that is not present in the network definition
            for key in network_params[i]:
                if key not in total_network_params[i] and 'args' not in total_network_params[i]:
                    raise ValueError(f'Error: network_params contains a key that is not present in the network definition: {key}')
            skip_check = 0
            if 'args' in total_network_params[i]:
                total_network_params[i].pop('args', None); skip_check = 1
            if 'kwargs' in total_network_params[i]:
                total_network_params[i].pop('kwargs', None); skip_check = 1
            l1 = len(total_network_params[i])
            total_network_params[i].update(network_params[i])
            total_network_params[i] = self.__sort_dictionary(total_network_params[i])
            if skip_check == 0:
                assert len(total_network_params[i]) == l1, 'Error: network_params contains a key that is not present in the network definition'
        network_params = total_network_params

        def remove_param(dictt,ignore):
            for key in ignore:
                if key in dictt.keys():
                    dictt.pop(key, None)
            for key in dictt.keys():
                if type(dictt[key]) is dict:
                    dictt[key] = remove_param(dictt[key],ignore)
            return dictt
        
        opt_class_str = []; total_opt_params = []
        for i,(opt,skip) in enumerate(zip(opt_fun,ignore_optimizer_params)):
            try:
                opt_class_str.append( inspect.getsource(opt) )
            except:
                raise ValueError('Error: opt_fun must be a function class not the defined optimizer with input parameters (e.g. it should be just "Optimizer", and not "Optimizer(params)" ).')
            total_opt_params.append( self.__initial_class_params_retriever(opt) )
            l2 = len(total_opt_params[-1])
            if 'best' in optimizer_params[i]:
                total_opt_params[-1] = optimizer_params[i]
            else:
                for key in optimizer_params[i]:
                    if key not in total_opt_params[-1]:
                        raise ValueError(f'Error: optimizer_params contains a key that is not present in the optimizer definition: {key}')
                    elif inspect.isclass(type(optimizer_params[i][key])):
                        if key+'_params' in optimizer_params[i]:
                            total_opt_params2 = self.__initial_class_params_retriever(optimizer_params[i][key])
                            ll1 = len(total_opt_params2)
                            total_opt_params2.update(optimizer_params[i][key+'_params'])
                            optimizer_params[i][key+'_params'] = total_opt_params2
                            assert len(optimizer_params[i][key+'_params']) == ll1, f"Error: parameters {key+'_params'} contains a key that is not present in {key} definition"

                total_opt_params[-1].update(optimizer_params[i])
                total_opt_params[-1] = self.__sort_dictionary(total_opt_params[-1])
                assert len(total_opt_params[-1]) == l2, 'Error: optimizer_params contains a key that is not present in the optimizer definition'
                total_opt_params[-1] = remove_param(total_opt_params[-1],skip)
                

        optimizer_params = copy.deepcopy(total_opt_params)

        try:
            loss_fun = loss_fun.__class__
        except:pass
        loss_class_str = inspect.getsource(loss_fun)
        loss_params = self.__initial_class_params_retriever(loss_fun)
        l3 = len(loss_params)
        loss_params.update(loss_params)
        loss_params = self.__sort_dictionary(loss_params)
        assert len(loss_params) == l3, 'Error: loss_params contains a key that is not present in the loss definition'
        loss_fun = loss_fun(**loss_params)

        return dataset, mb_size, opt_fun, optimizer_name, opt_class_str, optimizer_params, pretraining_status, network_fun, network_class_str, network_params, overlap_ratio, loss_fun, loss_params, loss_class_str


    def __make_keys_list(self, dataset=None, mb_size=None, opt_fun=None, optimizer_params=None, pretraining_status=None, overlap_ratio=None, network_fun=None, network_params=None, loss_fun=None, loss_params=None, 
                         ignore_optimizer_params=None, **kwargs):       
        '''
        Creates a list of keys from the given parameters. Each key is a dictionary representing a test.
        '''        
        dataset1, mb_size1, opt_fun1, optimizer_name1, opt_class_str1, optimizer_params1, pretraining_status1, network_fun1, network_class_str1, network_params1, overlap_ratio1, \
        loss_fun1, loss_params1, loss_class_str1 \
        = self.__input_params_check(dataset, mb_size, opt_fun, optimizer_params, 
            pretraining_status, overlap_ratio, network_fun, network_params, loss_fun, loss_params, ignore_optimizer_params)

        keys = []
        for mb in mb_size1:
            for o_r in overlap_ratio1:
                for i,nn in enumerate(network_class_str1):
                    for j,f in enumerate(opt_fun1):
                        key={'dataset':dataset1, 'opt_fun':f, 'optimizer_name':optimizer_name1[j], 'optimizer_params':optimizer_params1[j], 'opt_class_str':opt_class_str1[j], 'mb_size':mb, 
                             'pretraining_status':pretraining_status1, 'network_function':network_fun1[i], 'network_class_str':nn, 'network_input_params':network_params1[i], 'overlap_ratio':o_r, 
                            'loss_fun':loss_fun1, 'loss_params':loss_params1, 'loss_class_str':loss_class_str1, 'loss_name':loss_fun1.__class__.__name__}
                        keys.append(key)

        if self.comments:
            print(f'(__make_keys_list) Number of tests to perform: {len(keys)}')
        
        return keys
    
    def __compare_optimizers(self, keys=None, trials=10, epochs=50, dataset=None, optimizer_name=None, optimizer_params=None, mb_size=None, seed=None, pretraining_status=None, network_class_str=None, overlap_ratio=None):
        pass # TODO: allow for list of keys, retrieve results (or auto compute them) and plot and/or print them
        # USE THIS:    InteractivePlot(**kwargs)

    def __initial_class_params_retriever(self,fun):
        '''
        Retrieves the initial parameters of the network/optimizer from the string that defines its class.     
        TODO: Make this more adaptive with respect to the number of spaces.. maybe through RegEx ( "):" and ") :" are different)  
        '''
        if 'class' in str(type(fun)):
            fun2 = fun.__init__
        else:
            fun2 = fun

        signature = inspect.signature(fun2)
        parameters = signature.parameters
        vars2 = {}
        for name, param in parameters.items():
            if 'empty' in str(param.default):
                vars2[name] = ''
            else:
                vars2[name] = param.default

        if 'self' in vars2.keys():
            vars2.pop('self', None)

        return self.__sort_dictionary(vars2)
    
    def __sort_dictionary(self, dicti):
        '''
        Sorts a dictionaries and dictionaries inside dictionaries.
        '''
        for key in dicti.keys():
            if isinstance(dicti[key], dict):
                dicti[key] = self.__sort_dictionary(dicti[key])
        return {k: v for k, v in sorted(dicti.items(), key=lambda item: item[0])}
    

    def get(self, dataset, mb_size, opt_fun, optimizer_params, network_fun, loss_function, ignore_optimizer_params=['params'], 
            loss_params={}, network_params={}, overlap_ratio=0, trials=10, epochs=50, pretraining_status=0, IGNORED_FIELDS=None):
        '''
        Returns the required results, if they are not in the dataframe, then it computes them.
        TODO: Improve explanation

        IGNORED_FIELDS : Is a list of fields that will be ignored when looking for the results. E.g. if you want to retrieve the results of all the tests with the same optimizer name but different optimizer class string (maybe due to updated code or differenty pytorch version)
        '''
        if self.comments and network_params=={}:
            print('(get) network_params is empty, using default parameters.')
        if self.comments and loss_params=={} :
            print('(get) loss_params is empty, using default parameters.')
        if self.comments and len(ignore_optimizer_params)==0:
            print(f'(get) ignore_optimizer_params is empty. "params" should be in it.')

        assert isinstance(dataset, str), "Error: dataset must be a string."
        assert isinstance(mb_size, list) or isinstance(mb_size, int) or isinstance(mb_size, str), "Error: mb_size must be a list or an int/str  (only 'full' is an available string)."
        assert isinstance(opt_fun, list) or 'class' in str(type(opt_fun)), "Error: opt_fun must be a list or a function class."
        assert isinstance(optimizer_params, list) or isinstance(optimizer_params, dict), "Error: optimizer_params must be a list or a dictionary."
        assert isinstance(pretraining_status, int) or isinstance(pretraining_status, float), "Error: pretraining_status must be a float/integer."
        assert isinstance(network_fun, list)  or 'class' in str(type(network_fun)), "Error: network_fun must be a list or a string."
        assert isinstance(overlap_ratio, list) or isinstance(overlap_ratio, float) or isinstance(overlap_ratio, int) or overlap_ratio<0 or overlap_ratio>=1, "Error: overlap_ratio must be a list or a float/integer between 0 and 1."
        assert isinstance(network_params, list) or isinstance(network_params, dict), "Error: network_params must be a list or a dictionary."
        assert 'class' in str(type(loss_function)), "Error: loss_fun must be a function class."
        assert isinstance(loss_params, dict), "Error: loss_params must be a dictionary."
        assert isinstance(ignore_optimizer_params, list), "Error: ignore_optimizer_params must be a list."
        assert type(epochs) is int, 'Error: epochs has to be an integer'
        assert type(trials) is int, 'Error: trials has to be an integer'
        
        dict_of_keys = self.__make_keys_list(dataset=dataset, mb_size=mb_size, opt_fun=opt_fun, optimizer_params=optimizer_params, pretraining_status=pretraining_status, 
                                             overlap_ratio=overlap_ratio, network_fun=network_fun, network_params=network_params,loss_fun=loss_function, loss_params=loss_params, 
                                             ignore_optimizer_params=ignore_optimizer_params)

        global_df = {}
        global_df_plot = {}
        for index,key_dict in enumerate(dict_of_keys):
            assert key_dict['pretraining_status'] == 0, 'Error: Not correctly implemented yet'
            Trials = trials
            global_df_plot[index] = {'accuracy': pd.DataFrame(), 'loss': pd.DataFrame(), 'time': pd.DataFrame()}
            if type(key_dict['optimizer_params']) is not dict and 'best' in key_dict['optimizer_params']:
                best=key_dict['optimizer_params']
                del key_dict['optimizer_params']
                if 'median' in best:
                    mode='median'
                elif 'alpha' in best:
                    mode='0.05_mean'
                    best.replace(mode,'')
                else:
                    pattern = r'[-_]?(\d+\.\d+)[-_]mean'
                    match = re.search(pattern, best)
                    if match:
                        mode=match.group(1) + "_mean"
                    else:
                        mode='mean'
                if 'accuracy' in best:
                    measure='accuracy'
                elif 'loss' in best:
                    measure='loss'
                elif 'time' in best:
                    raise ValueError('Error: time is not a valid measure')
                else:
                    measure='accuracy'
                    
                pattern = r'\d+\.\d+|\d+'
                epochs = re.findall(pattern, best)
                if len(epochs) == 0:
                    epochs=20
                else:
                    for i in range(len(epochs)):
                        epochs[i] = float(epochs[i])
                    epochs = max(epochs)
                    if int(epochs)==float(epochs):
                        epochs=int(epochs)
                    else:
                        epochs=20

                print(f'Finding best params in {epochs} epochs for {key_dict["optimizer_name"]} with respect to the {mode} of {measure}')
                if self.enable_hints:
                    print('\nHint: If you want to change the number of epochs, the mode or the measure, you can do it by changing the optimizer_params string \
                          \n(e.g. "best_of_50epochs_0.03_mean_accuracy" or "best_in_100_median_loss").\
                          \nNote: Large integer numbers will be considered as epochs and floats as alpha. \
                          \nAvailable modes are median,mean,alpha_mean (0.05-mean,0.01-mean,...) -> does not consider best/worst 5% or 1%.\
                          \nAvailable measures are accuracy,loss,time.\n')
                key_dict['optimizer_params'] = self.__find_optimal_params(epochs, mode=mode, measure=measure, IGNORED_FIELDS=IGNORED_FIELDS,**key_dict)

            row_ids,seeds = self.__select_results(IGNORED_FIELDS,**key_dict)
            if len(row_ids) == 0: # 0 trials performed
                df = {}
                for i in range(Trials):
                    df[str(i*self.seed_interval)] = {}
                assert loss_function is not None, 'Error: Trying to run the required tests but loss_function is not specified.'
                df = self.__do_tests(df, key_dict, epochs=epochs, criterion=loss_function)
                # global_df_plot[index] = df_plot
                for field in ['accuracy','loss','time']:
                    global_df_plot[index][field] = pd.concat([df[d][field] for d in df],axis=1)

            else: # keys = [key]
                df = {};  #Final dataframe with x trials
                df2 = {}; #Dataframe with trials to perform
                for data,seed in zip(row_ids,seeds):
                    epo = len(data[1])-1
                    if epo >= epochs:
                        df[seed] = data[1].head(epochs+1) # Cuts the DataFrame to the first "epochs" rows
                        Trials -= 1
                    elif data[2] is not None and data[3] is not None: # optimizer and net state are saved so we can continue from where we left and add more epochs
                        df2[seed] = data[1]
                        Trials -= 1
                    else: # optimizer and net state are not saved so we have to start from scratch
                        pass
                    if Trials == 0:
                        break

                if Trials > 0:
                    for i in range(Trials):
                        df2[str((i+1)*self.seed_interval+max(seeds))] = {}

                if len(df2) > 0:
                    assert loss_function is not None, 'Error: Trying to run the required tests but loss_function is not specified.'
                    df2 = self.__do_tests(df2, key_dict, epochs=epochs, criterion=loss_function)
                    df.update(df2)

                for field in ['accuracy','loss','time']:
                    global_df_plot[index][field] = pd.concat([df[d][field] for d in df],axis=1)

            if self.DEBUG:
                print('("get" function) Printing df')
                print(df)

            global_df[index] = df
        return global_df, global_df_plot, dict_of_keys # Dataframe as dictionary with seed as keys and accuracy,loss,times inside each key
    
    def merge(self, df2,subkey=''):
        '''
        Merges two dataframes into one. The second dataframe "df2" will be inclueded in the current (self) one.
        Note: subkey is a list of strings that has to be in every key that you want to merge. So you can, e.g., transfer SGD results only or Adam results related to MNIST and not CIFAR, ...
        '''
        if type(subkey) is not list:
            subkey = [subkey]
        c=0
        for k in df2.df['results'].keys():
            complete_key = k.split('%')
            Net_nr = complete_key[self.key_fields.index('network_class_str')]
            complete_key[self.key_fields.index('network_class_str')] = df2.df['net_code_list'][int(Net_nr)]
            key_shortened = self.__key_shortener('%'.join(complete_key))
            if 'MNIST' in key_shortened and 'full' in key_shortened:
                key_shortened = key_shortened.replace('full','60000')
            if 'MNIST' in key_shortened and '60000' in key_shortened and float(key_shortened.split('%')[-1])!=0:
                key_shortened = '%'.join(key_shortened.split('%')[:-1]+ ['0.0'])

            if key_shortened not in self.df['results'].keys() and all([s in key_shortened for s in subkey]):
                c+=1
                self.df['results'][key_shortened] = df2.df['results'][k]
                if self.save_optimizer_and_net_status:
                    self.df['net_state'][key_shortened] = df2.df['net_state'][k]
                    self.df['optimizer_fun'][key_shortened] = df2.df['optimizer_fun'][k]
                    for lvl in self.pretraining_lvl:
                        self.df[f"net_state_{lvl}"].update(df2.df[f"net_state_{lvl}"])
            elif not (key_shortened not in self.df['results'].keys()) and all([s in key_shortened for s in subkey]):
                for seed in df2.df['results'][k].keys():
                    if seed not in self.df['results'][key_shortened].keys():
                        c+=1
                        self.df['results'][key_shortened][seed] = df2.df['results'][k][seed]
                        if self.save_optimizer_and_net_status:
                            self.df['net_state'][key_shortened][str(seed)] = df2.df['net_state'][k][str(seed)]
                            self.df['optimizer_fun'][key_shortened][str(seed)] = df2.df['optimizer_fun'][k][str(seed)]
                            for lvl in self.pretraining_lvl:
                                self.df[f"net_state_{lvl}"][str(seed)] = df2.df[f"net_state_{lvl}"][k][str(seed)]
        if subkey == '':
            print('Merged',c,'keys')
        else:
            print('Merged',c,'keys with subkey',subkey)

    
    # Mode: "mean", "median", "alpha_mean", "0.05-mean" (alpha_mean with alpha=0.05)
    def plot(self, dataset, mb_size, opt_fun, optimizer_params, network_fun, loss_fun, ignore_optimizer_params=['params'], 
            loss_params={}, network_params={}, overlap_ratio=0, trials=10, epochs=50, pretraining_status=0, mode="mean", SAVE=False,
            OPT_NAME = None, plot_type=[['accuracy','loss'],['time']], # accuracy, loss in one plot and time in another, you can only use "accuracy", "loss" and "time" words
            IGNORED_FIELDS=None, **kwargs):
        '''
        Plots the required results, if they are not in the dataframe, then it computes them.

        dictionary_of_input_variables_related_to_plotting:{
            text_size=14,                   # Size of the text in the plot
            legend_text_size=14,            # Size of the legend text
            title_text_size=16,             # Size of the title text
            linewidth=2,                    # Width of the lines in the plot
            show_variance=False,            # If True, shows the variance of the results (dashed lines)
            show_min_max=False,             # If True, shows the min and max of the results (dotted lines)
            unlink_plots_interactive=True   # If False, when you click on a legend it will show/hide the corresponding graph on every plot. Otherwise only on the plot you clicked.
            accuracy_ylim=[0, 100],         # Y limits of the accuracy plot  (e.g. [10, 100] or even 'best' to use adaptive mode)
            accuracy_xlim=[0, epochs],      # X limits of the accuracy plot
            loss_ylim=[0, 100],             # Y limits of the loss plot
            loss_xlim=[0, epochs],          # X limits of the loss plot
            time_ylim=[0, 100],             # Y limits of the time plot
            time_xlim=[0, epochs],          # X limits of the time plot
            Personal_Title=None,            # Personal title to add to the plot
            accuracy_scale='linear',        # Scale of the accuracy plot (e.g. 'log')
            loss_scale='linear',            # Scale of the loss plot (e.g. 'log')
        }
        '''
        text_size = kwargs.get('text_size',14)
        legend_text_size = kwargs.get('legend_text_size',14)
        title_text_size = kwargs.get('title_text_size',16)
        linewidth = kwargs.get('linewidth',2)
        show_variance = kwargs.get('show_variance',False)
        show_min_max = kwargs.get('show_min_max',False)
        unlink_plots_interactive = kwargs.get('unlink_plots_interactive',False)
        if self.regression:
            accuracy_ylim = kwargs.get('accuracy_ylim',None)
        else:
            accuracy_ylim = kwargs.get('accuracy_ylim','best')
        accuracy_xlim = kwargs.get('accuracy_xlim',[0, epochs])
        loss_ylim = kwargs.get('loss_ylim',None)
        loss_xlim = kwargs.get('loss_xlim',[0, epochs])
        time_ylim = kwargs.get('time_ylim',None)
        time_xlim = kwargs.get('time_xlim',[1, epochs]) #removing 0-th epoch because its 0. Training is not started yet
        Personal_Title = kwargs.get('Personal_Title',None)
        accuracy_scale = kwargs.get('accuracy_scale','linear')
        loss_scale = kwargs.get('loss_scale','linear')
        legend_position = kwargs.get('legend_position','center right')
        TITLE = kwargs.get('TITLE',True) # Enable, disable automatic title (or set TTTLE as str to set a custom title) 




        def perc(x):
            if x*100 == int(x*100):
                return str(int(x*100))+'%'
            else:
                return str(x*100)+'%'
            
        def classify_nn_type(nn_string):
            nn_string = nn_string.lower()  # Convert to lowercase for case-insensitive matching
            if "conv" in nn_string:
                return "CNN"  # Convolutional Neural Network
            elif "lstm" in nn_string or "gru" in nn_string or "rnn" in nn_string:
                return "RNN"  # Recurrent Neural Network
            elif "autoencoder" in nn_string or "vae" in nn_string:
                return "Autoencoder"  # Autoencoder
            elif "siamese" in nn_string:
                return "Siamese Network"  # Siamese Network
            elif "gan" in nn_string:
                return "GAN"  # Generative Adversarial Network
            elif "linear" in nn_string:
                return "FCNN"  # Fully Connected Neural Network
            # Add more network types here as needed
            else:
                return "Other NN"  # Other type of neural network
            
        def different_params(dict_list,optimizer_name,i):
            #This function compares every parameter dictionary in the list "dict_list" with the i-th one and returns the different parameters only. 
            #The parameters which do not change are not shown.
            if len(dict_list)==1:
                return ''
            else:
                s = ''
                if type(dict_list[i]) is not dict: #should be string ('best')
                    return 'best'
                else:
                    for key in dict_list[i]:
                        if not all([key in n_params for j,n_params in enumerate(dict_list) if optimizer_name[i]==optimizer_name[j]]) or not all([dict_list[i][key]==n_params[key] for j,n_params in enumerate(dict_list) if optimizer_name[i]==optimizer_name[j]]):
                            # if type(dict_list[i][key]) is dict:
                            #     s += key+':{'
                            #     for key2 in dict_list[i][key]:
                            #         if not all([key2 in n_params[key] for j,n_params in enumerate(dict_list) if optimizer_name[i]==optimizer_name[j]]) or not all([dict_list[i][key][key2]==n_params[key][key2] for j,n_params in enumerate(dict_list) if optimizer_name[i]==optimizer_name[j]]):
                            #             s += key2+':'+str(dict_list[i][key][key2])+','
                            #     s = s[:-1]+'},'
                            # else:
                                s += key+':'+str(dict_list[i][key])+','
                    return s[:-1]


        dataset, mb_size, opt_fun, optimizer_name, _, optimizer_params, pretraining_status, network_fun, network_class_str, network_params, overlap_ratio, \
        loss_fun, loss_params, loss_class_str \
        = self.__input_params_check(dataset, mb_size, opt_fun, optimizer_params, 
            pretraining_status, overlap_ratio, network_fun, network_params, loss_fun, loss_params, ignore_optimizer_params)
                
        assert OPT_NAME is None or len(OPT_NAME) == len(opt_fun), 'Error: OPT_NAME must have the same length as opt_fun'
        if mode not in ['mean','median','alpha_mean']:
            if '-' in mode:
                val,mean = mode.split('-')
            elif '_' in mode:
                val,mean = mode.split('_')
            else:
                raise ValueError('Error: mode must be "mean", "median", "alpha_mean" or e.g. "0.05-mean" (alpha_mean with alpha=0.05, or different values)')
            try:
                assert float(val)<1 and float(val)>0, 'Error: alpha-mean must be between 0 and 1'
            except:
                raise ValueError('Error: alpha-mean must have alpha as a float between 0 and 1')



        if self.enable_hints:
            print('(Hint) You can set the parameters to "best" to automatically find the best parameters for the given dataset and optimizer.')
        
        optimizer_params2 = copy.deepcopy(optimizer_params)
        df,df_plot,dict_of_keys = self.get(dataset, mb_size, opt_fun, optimizer_params2, network_fun, loss_fun, ignore_optimizer_params, 
                                loss_params, network_params, overlap_ratio, trials, epochs, pretraining_status, IGNORED_FIELDS)

        network_names = [n.__name__ for n in network_fun]
        for k in dict_of_keys:
            if k['optimizer_params'] == 'best':
                del k['optimizer_params']
                k = self.__find_optimal_params(mode=mode, measure='accuracy', IGNORED_FIELDS=IGNORED_FIELDS, **k) 

        for k in dict_of_keys:
            k['optimizer_name'] = k['opt_fun'].__name__ #The previous key was with "lower()" case

        TOT_SAMPLES = self.__load_data(dataset=dataset, data_dir=self.data_dir, TOT_SAMPLES=True)
        fig = []; ax = []; df2 = {}
        c = -1
        for res in plot_type:
            if unlink_plots_interactive:
                ax = []
            if len(res)<=1:
                Fig, Ax = plt.subplots()
            else:
                if c==-1:
                    default_figsize_original = copy.deepcopy(plt.rcParams["figure.figsize"])
                default_figsize = plt.rcParams["figure.figsize"]
                default_figsize[0] = default_figsize[0]*1.2
                Fig, Ax = plt.subplots(figsize=default_figsize)
                Fig.subplots_adjust(right=float(0.8))
            fig.append(Fig); ax.append(Ax)
            c += 1
            plt.figure(fig[c])
            p = [[] for _ in range(len(mb_size)*len(overlap_ratio)*len(network_class_str)*len(opt_fun))] # total amount of plots
            ii_check=0
            for ii,r in enumerate(res):
                pc = -1# p coutner
                df2[r] = {}; opt = []
                for mb in mb_size:
                    for o_r in overlap_ratio:
                        # This is needed to define a name for legend or title depending on what is changing in the plot
                        mb_amount = int(np.ceil(TOT_SAMPLES/mb))
                        if len(mb_size)==1 and len(overlap_ratio)==1:
                            MB = '';OL = '' # Legend
                            MB2 = 'mb #'+str(mb_amount)+f'(size {int(mb+(mb_amount-1)*o_r*mb)})'  # Title
                            if mb_amount>1:
                                OL2 = 'overlap '+perc(o_r) # Title
                            else: 
                                OL2 = ''
                        elif len(mb_size)==1 and len(overlap_ratio)!=1:
                            MB = ''; OL2 = ''
                            if mb_amount>1:
                                OL = ',overlap '+perc(o_r)
                            else:
                                OL = ''
                            MB2 = 'mb #'+str(mb_amount)
                        elif len(mb_size)!=1 and len(overlap_ratio)==1:
                            MB2 = ''; OL = ''
                            MB = ',mb #'+str(mb_amount)
                            if mb_amount>1:
                                OL2 = 'overlap '+perc(o_r) # Title
                            else: 
                                OL2 = ''
                        else:
                            MB = ',mb #'+str(mb_amount)
                            if mb_amount>1:
                                OL = ',overlap '+perc(o_r)
                            else:
                                OL = ''
                            MB2 = ''; OL2 = ''

                        for i,nn in enumerate(network_class_str):
                            # This is needed to define a name for legend or title depending on what is changing in the plot
                            NN_name = classify_nn_type(nn)
                            if len(network_class_str)==1 or all([n==nn for n in network_class_str]):
                                if len(network_params)==1 or all([n==network_params[0] for n in network_params]):
                                    NN = ''; NN2 = NN_name
                                else:
                                    rdp = different_params(network_params, network_names, i)
                                    NN = ','+NN_name+'('+rdp+')'
                                    NN2 = ''
                            else:
                                if len(network_params)==1 or all([n==network_params[0] for n in network_params]):
                                    NN = ','+network_fun.__name__
                                    NN2 = ''
                                else:
                                    rdp = different_params(network_params, network_names, i)
                                    NN = ','+network_fun.__name__+'('+rdp+')'
                                    NN2 = ''

                            for j,f in enumerate(opt_fun):
                                # This is needed to define a name for legend or title depending on what is changing in the plot
                                if len(opt_fun)==1 or all([n==f for n in opt_fun]):
                                    if len(optimizer_params)==1 or all([n==optimizer_params[0] for n in optimizer_params]):
                                        OPT = ''; OPT2 = optimizer_name[j]
                                    else:
                                        rdp = different_params(optimizer_params, optimizer_name, j)
                                        OPT = optimizer_name[j]+'('+rdp+')' if rdp!='' else optimizer_name[j]
                                        OPT2 = ''
                                else:
                                    if len(optimizer_params)==1 or all([n==optimizer_params[0] for n in optimizer_params]):
                                        OPT = optimizer_name[j]; OPT = optimizer_name[j]; OPT2 = ''
                                    else:
                                        rdp = different_params(optimizer_params, optimizer_name, j)
                                        OPT = optimizer_name[j]+'('+rdp+')' if rdp!='' else optimizer_name[j]
                                        OPT2 = ''
                                for k,key in enumerate(dict_of_keys):
                                    if mb!=key['mb_size'] or nn!=key['network_class_str'] or f!=key['opt_fun'] or optimizer_name[j].lower()!=key['optimizer_name'].lower() \
                                        or (key['optimizer_params']!=optimizer_params[j] and 'best' not in optimizer_params[j]) or key['network_input_params']!=network_params[i]:
                                        continue
                                    pc += 1
                                    if OPT_NAME is not None:
                                        opt.append(OPT_NAME[j]) # Legend name
                                    else:
                                        if OPT+NN+MB+OL == '':
                                            opt.append(optimizer_name[j])
                                        else:
                                            opt.append(OPT+NN+MB+OL) # Legend name

                                    df2[r] = pd.DataFrame()
                                    df2[r]['mean'] = df_plot[k][r].mean(axis=1)
                                    df2[r]['min'] = df_plot[k][r].min(axis=1)
                                    df2[r]['max'] = df_plot[k][r].max(axis=1)
                                    df2[r]['std'] = df_plot[k][r].std(axis=1)
                                    df2[r]['median'] = df_plot[k][r].median(axis=1)
                                    if '-' in mode: # Set the significance level (alpha) for alpha-mean
                                        alpha = float(mode.split('-')[0])
                                        mode = 'alpha_mean'
                                    elif 'alpha' not in locals():
                                        alpha = 0.05 
                                    df2[r]['alpha_mean'] = df_plot[k][r].apply(lambda row: np.mean(row[row >= np.percentile(row, alpha * 100)]), axis=1)

                                    if mode == 'mean' or mode == 'median':
                                        data_y = df2[r][mode]
                                    elif mode == 'alpha_mean':
                                        data_y = df2[r]['alpha_mean']

                                    epochs = df2[r]['mean'].index
                                    if ii!=ii_check:
                                        ax.append(ax[-ii].twinx())
                                        ax[-1].spines['right'].set_position(('axes', 1.0+0.15*(ii-1)))
                                        ii_check=ii

                                    # Plot data
                                    p1,=ax[-1].plot(epochs, data_y, label=opt[-1]);P=[p1]
                                    color = p1.get_color()
                                    if show_variance:
                                        p2,=ax[-1].plot(epochs, data_y-df2[r]['std'], linewidth=linewidth, linestyle='--', color=color, label=opt[-1])
                                        p3,=ax[-1].plot(epochs, data_y+df2[r]['std'], linewidth=linewidth, linestyle='--', color=color, label=opt[-1])  
                                        P.append(p2); P.append(p3)
                                    if show_min_max:
                                        p4,=ax[-1].plot(epochs, df2[r]['min'], linestyle='dotted', linewidth=linewidth, color=color, label=opt[-1])
                                        p5,=ax[-1].plot(epochs, df2[r]['max'], linestyle='dotted', linewidth=linewidth, color=color, label=opt[-1])
                                        p6 =ax[-1].fill_between(epochs, df2[r]['min'], df2[r]['max'], color=color, alpha=0.3)
                                        P.append(p4); P.append(p5); P.append(p6)
                                    p[pc].extend(P)
                if r == 'accuracy':
                    ax[-1].set_yscale(accuracy_scale)
                elif r == 'loss':
                    ax[-1].set_yscale(loss_scale)
                # horizontal lines
                if r == 'accuracy':
                    y_values = list(range(0,110,10))  # Specify the y-values for the horizontal lines
                    ax[-1].hlines(y_values, min(epochs), max(epochs), colors='black', linestyles='dashed', linewidths=0.5)
                    if accuracy_ylim == 'best':
                        artists = ax[-1].get_children()
                        y_data = []
                        for artist in artists:
                            if isinstance(artist, plt.Line2D):
                                y_data = np.concatenate([y_data,artist.get_ydata()])
                        t1 = np.percentile(y_data, 10); t1 = np.floor(t1/10)*10
                        t2 = max(y_data); t2 = np.ceil(t2/10)*10
                        accuracy_ylim = [t1,min(t2,100)]
                    ax[-1].set_ylim(accuracy_ylim)
                    ax[-1].set_xlim(accuracy_xlim)
                elif r == 'loss':
                    ax[-1].set_ylim(loss_ylim)
                    ax[-1].set_xlim(loss_xlim)
                elif r == 'time':
                    ax[-1].set_ylim(time_ylim)
                    ax[-1].set_xlim(time_xlim)

                ax[-1].tick_params(axis='both', which='major', labelsize=text_size)
                plt.ylabel(f'Avg. {r}', fontsize=text_size)

            # legend = ax[-1].legend([tuple(p[ii]) for ii in range(len(opt))],[opt[ii] for ii in range(len(opt))],loc='center right', shadow=False, fontsize=legend_text_size, framealpha=0.5)
            legend = ax[-1].legend([tuple(p[ii]) for ii in range(len(opt))],[opt[ii] for ii in range(len(opt))],loc=legend_position, shadow=False, fontsize=legend_text_size, framealpha=0.5)
            plot_title = ' - '.join([t for t in [NN2,OPT2,MB2,OL2] if t!=''])
            if Personal_Title is not None:
                plot_title = Personal_Title 
            
            # setting xlabel:
            if TITLE:
                plt.title(plot_title if TITLE is True else TITLE, fontsize=title_text_size)
            ax[0].set_xlabel('Epoch', fontsize=text_size)
            
            InteractivePlot2(legend,ax)
            # Saving
            path = '/'.join(self.results_filename.split('/')[:-1])
            if OPT_NAME is None:
                name = f'DF_{r}_{dataset}_mb{mb}_o{overlap_ratio}_{"-".join([o.split("{")[0] for o in opt])}'
            else:
                name = f'DF_{r}_{dataset}_mb{mb}_o{overlap_ratio}_{"-".join(OPT_NAME)}'
            if SAVE:
                if SAVE is not True: # SAVE is a path
                    if '.png' in SAVE:
                        SAVE = SAVE.replace('.png','')
                    name = SAVE + f'_{"_".join(res)}.png'
                    plt.savefig(f'{name}', bbox_inches='tight', dpi=300)
                    print(f'{SAVE} Plot is saved')
                else:
                    try:
                        plt.savefig(f'{path}/{name}.png', bbox_inches='tight', dpi=300)
                        print(f'{name}.png Plot is saved in path {path}')
                    except Exception as e:
                        print(f'{name}.png Plot is NOT saved in path {path}. {e}')

        plt.show()
        plt.rcParams["figure.figsize"] = default_figsize_original
        return



if __name__=='__main__':
    # Run some tests here

    name = lambda x: 'G:\\.shortcut-targets-by-id\\1S1Znw8az1dTTW6dO8Ksdp9j5avi6DX7L\\ML_Project\\MultiscAI\\ML2\\results\\sam\\main_daint_test_copy_'+str(x)+'.db'
    a = Power_Dataframe(results_filename=r"G:\.shortcut-targets-by-id\1S1Znw8az1dTTW6dO8Ksdp9j5avi6DX7L\ML_Project\MultiscAI\ML2\results\sam\ASD.db", sequential=True)
    for i in range(0,50):
        b=Power_Dataframe( results_filename = name(i) , sequential = True )
        a.merge_databases(b)


    args = parse_args()
    DB_PATH = args.db_path
    # a = Power_Dataframe(results_filename="./results/TEMP.db", sequential=True)
    # a = Power_Dataframe(results_filename=DB_PATH, sequential=True, HOST='localhost', PORT=9999)
    a = Power_Dataframe(results_filename="./results/TEMP1.db", sequential=True)
    # a2 = Power_Dataframe(results_filename="./results/TEMP2.db", sequential=True)
    # a.merge(a2)
    # a.__del_db__()

    # a.merge_databases(a2)
    
    net = NetMNIST2
    opt_params=[]
    # for comb in itertools.product(['MNIST'], ['SGD','ADAM'], [{'lr':0.01,'momentum':0.9},{'lr':0.01,'momentum':0.8},{'lr':0.1,'momentum':0.9}], [10000]):
    for comb in itertools.product(['MNIST'], ['SGD','ADAM'], [{'lr':0.01},{'lr':0.02},{'lr':0.03}], [10000]): 
        if comb[1]=='ADAM': 
            opt_fun = torch.optim.Adam
        elif comb[1]=='SGD':
            opt_fun = torch.optim.SGD
        else:
            opt_fun = APTS_W
        opt_params.append(comb[2])
        network_params = {'hidden_sizes':[64,32]}
        df=a.get(dataset=comb[0], mb_size=comb[3], opt_fun=opt_fun, optimizer_params=comb[2], ignore_optimizer_params=['params'], network_fun=net, network_params=network_params, 
                 loss_function=nn.CrossEntropyLoss(), trials=6, epochs=7, pretraining_status=0, overlap_ratio=0)
    print('done')

    # opt_params='best6'
    # a.plot(comb[0], mb_size=comb[3], opt_fun=opt_fun, optimizer_params=[{'lr':0.01},{'lr':0.02},{'lr':0.03}], network_fun=net, loss_fun=nn.CrossEntropyLoss(), ignore_optimizer_params=['params'], 
    #         loss_params={}, network_params={}, overlap_ratio=0, trials=10, epochs=50, pretraining_status=0, mode="mean", SAVE=False,
    #         OPT_NAME = None, text_size=14, legend_text_size=14, title_text_size=16, linewidth=2, show_variance=False,
    #         plot_type=[['accuracy','loss'],['time']] # accuracy, loss in one plot and time in another
    #       )
    print('done')


