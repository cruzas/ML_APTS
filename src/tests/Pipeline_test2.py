# import torch
# import torch.nn as nn
# import torch.multiprocessing as mp
# from queue import Queue



# def main(rank=None, master_addr=None, master_port=None, world_size=None):
#     # Define the functions f and g as PyTorch modules
#     class F(nn.Module):
#         def __init__(self):
#             super(F, self).__init__()
#             self.layer = nn.Linear(100, 50000)

#         def forward(self, x):
#             return self.layer(x)

#     class G(nn.Module):
#         def __init__(self):
#             super(G, self).__init__()
#             self.layer = nn.Linear(50000, 20)

#         def forward(self, x):
#             return self.layer(x)
        
#     x = torch.randn(10000, 100)  # 10000 samples, 100 features each
#     # Initialize models and assign to GPUs
#     f = F().to('cuda:0')
#     g = G().to('cuda:1')
#     # Calculate chunk size and start index for each process
#     chunk_size = x.size(0) // world_size
#     start_idx = rank * chunk_size
#     end_idx = start_idx + chunk_size if rank != world_size - 1 else x.size(0)

#     # Select the chunk for this process
#     chunk = x[start_idx:end_idx].to(f'cuda:{rank}')

#     # Process the chunk by f
#     output_f = f(chunk)

#     # Send output to the next GPU (simulate data passing to g)
#     # Move output to the next GPU's memory if there are multiple GPUs available
#     next_gpu = (rank + 1) % world_size
#     output_f = output_f.to(f'cuda:{next_gpu}')

#     # Process by g on the next GPU
#     output_g = g(output_f)

#     # Print output details
#     print(f"Processed by G on GPU {next_gpu}: Output shape {output_g.shape}")


# # Main function to setup data and call run_pipeline
# if __name__ == '__main__':
#     torch.manual_seed(1)
#     world_size = torch.cuda.device_count()  
#     master_addr = 'localhost'
#     master_port = '12345'  
#     mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)












# This works <-----------------

import torch
import torch.nn as nn
from queue import Queue
import threading

class F(nn.Module):# Define the functions f and g as PyTorch modules
    def __init__(self):
        super(F, self).__init__()
        self.layer = nn.Linear(100, 50000)
    def forward(self, x):
        return self.layer(x)

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.layer = nn.Linear(50000, 1)
    def forward(self, x):
        return self.layer(x)

def pipeline_function(x, f, g, chunk_size=5000): # Pipeline function
    chunks = [x[i:i + chunk_size] for i in range(0, x.size(0), chunk_size)] # Split data into chunks
    f_to_g_queue = Queue() # Queues for inter-thread communication

    # Function to process chunks by f and send to g
    def process_by_f():
        for chunk in chunks:
            # Move chunk to GPU 0 and process
            processed_chunk = f(chunk.to('cuda:0'))
            f_to_g_queue.put(processed_chunk.to('cuda:1'))
    
    # Function to process chunks by g
    def process_by_g():
        processed_chunks = []
        for _ in range(len(chunks)):
            # Get processed chunk from f and apply g
            chunk = f_to_g_queue.get()
            processed_chunk = g(chunk)
            processed_chunks.append(processed_chunk.cpu())  # Move back to CPU or to another destination
            print("Processed chunk by G:", processed_chunk.shape)
    
    # Initialize threads
    f_thread = threading.Thread(target=process_by_f)
    g_thread = threading.Thread(target=process_by_g)

    # Start threads
    f_thread.start()
    g_thread.start()

    # Wait for threads to finish
    f_thread.join()
    g_thread.join()

# Example usage
if __name__ == '__main__':
    # Assume some random data
    x = torch.randn(50000, 100)  # 128 samples, 100 features each

    # Initialize models and move to appropriate GPUs
    f = F().to('cuda:0')
    g = G().to('cuda:1')

    # Call the pipeline function
    pipeline_function(x, f, g)
