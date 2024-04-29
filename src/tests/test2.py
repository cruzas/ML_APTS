from torch.multiprocessing import Process, set_start_method
import torch
import time

stream = [None]*2
# define streams for each GPU
for i in range(2):
    stream[i] = torch.cuda.Stream(device=i)
    
torch.cuda.synchronize()
def process(ii):
    global stream
    with torch.cuda.stream(stream[ii]):
        for i in range(110):
            print(f"IM HERE {ii}\n")

if __name__ == "__main__":
    set_start_method('spawn',force = True)
    start = time.time()
    p1 = Process(target = process, args = (0,))
    p2 = Process(target = process, args = (1,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    torch.cuda.synchronize()
    
# from torch.multiprocessing import Process, set_start_method
# import torch
# import time


# stream1 = torch.cuda.Stream()
# stream2 = torch.cuda.Stream()
# torch.cuda.synchronize()
# def process1():
#     global stream1
#     with torch.cuda.stream(stream1):
#         for i in range(110):
#             print("IM HERE 1\n")
#         # print(time.time(),"time in process 1")
#         # time.sleep(5)
# def process2():
#     global stream2
#     with torch.cuda.stream(stream2):
#         for i in range(110):
#             print("IM HERE 2\n")
#         # print(time.time(),"time in process 2")
#         # time.sleep(5)

# if __name__ == "__main__":
#     set_start_method('spawn',force = True)
#     start = time.time()
#     p1 = Process(target = process1)
#     p2 = Process(target = process2)
#     p1.start()
#     p2.start()
#     p1.join()
#     p2.join()
#     torch.cuda.synchronize()