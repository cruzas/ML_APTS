import torch, os 
import torch.multiprocessing as mp
import torch.distributed as dist
from pippy import pipeline, annotate_split_points, Pipe, SplitPoint
from pippy.PipelineStage import PipelineStage

def prepare_distributed_environment(rank=None, master_addr=None, master_port=None, world_size=None):
    device_id = 0
    if rank is None and master_addr is None and master_port is None and world_size is None: # we are on a cluster
        print(f'Should be initializing {os.environ["SLURM_NNODES"]} nodes')
        ## Execute code on a cluster
        os.environ["MASTER_PORT"] = "29501"
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NNODES"]
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = os.environ["SLURM_NODEID"]
        node_list = os.environ["SLURM_NODELIST"]
        master_node = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1"
        )
        os.environ["MASTER_ADDR"] = master_node
        print(f"Dist initialized before process group? {dist.is_initialized()}")
        dist.init_process_group(backend="nccl")
        print(f"Dist initialized after init process group? {dist.is_initialized()} with world size {dist.get_world_size()}")
    else: # we are on a PC
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port # A free port on the master node
        # os.environ['WORLD_SIZE'] = str(world_size) # The total number of GPUs in the distributed job
        # os.environ['RANK'] = '0' # The unique identifier for this process (0-indexed)
        # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo" # "nccl" or "gloo"
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    device_id = dist.get_rank()
    print(f"Device id: {device_id}")

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class MyNetworkBlock(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.lin = torch.nn.Linear(in_dim, out_dim)

        def forward(self, x):
            x = self.lin(x)
            x = torch.relu(x)
            return x


    class MyNetwork(torch.nn.Module):
        def __init__(self, in_dim, layer_dims):
            super().__init__()

            prev_dim = in_dim
            for i, dim in enumerate(layer_dims):
                setattr(self, f'layer{i}', MyNetworkBlock(prev_dim, dim))
                prev_dim = dim

            self.num_layers = len(layer_dims)
            # 10 output classes
            self.output_proj = torch.nn.Linear(layer_dims[-1], 10)

        def forward(self, x):
            for i in range(self.num_layers):
                x = getattr(self, f'layer{i}')(x)

            return self.output_proj(x)



    in_dim = 512
    layer_dims = [512, 1024, 256]
    mn = MyNetwork(in_dim, layer_dims).to(device)




    annotate_split_points(mn, {'layer0': SplitPoint.END,
                            'layer1': SplitPoint.END})

    batch_size = 32
    example_input = torch.randn(batch_size, in_dim, device=device)
    chunks = 4

    pipe = pipeline(mn, chunks, example_args=(example_input,))
    print(pipe)

    """
    ************************************* pipe *************************************
    GraphModule(
    (submod_0): PipeStageModule(
        (L__self___layer0_mod_lin): Linear(in_features=512, out_features=512, bias=True)
    )
    (submod_1): PipeStageModule(
        (L__self___layer1_mod_lin): Linear(in_features=512, out_features=1024, bias=True)
    )
    (submod_2): PipeStageModule(
        (L__self___layer2_lin): Linear(in_features=1024, out_features=256, bias=True)
        (L__self___output_proj): Linear(in_features=256, out_features=10, bias=True)
    )
    )

    def forward(self, arg0):
        submod_0 = self.submod_0(arg0);  arg0 = None
        submod_1 = self.submod_1(submod_0);  submod_0 = None
        submod_2 = self.submod_2(submod_1);  submod_1 = None
        return [submod_2]
    """

    # We are using `torchrun` to run this example with multiple processes.
    # `torchrun` defines two environment variables: `RANK` and `WORLD_SIZE`.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize distributed environment

    dist.init_process_group(rank=rank, world_size=world_size)

    # Pipeline stage is our main pipeline runtime. It takes in the pipe object,
    # the rank of this process, and the device.
    stage = PipelineStage(pipe, rank, device)

    # Input data
    x = torch.randn(batch_size, in_dim, device=device)

    # Run the pipeline with input `x`. Divide the batch into 4 micro-batches
    # and run them in parallel on the pipeline
    if rank == 0:
        stage(x)
    elif rank == world_size - 1:
        output = stage()
    else:
        stage()
        
        
if __name__ == '__main__':
    try:
        operative_system = os.environ['OS']
    except:
        operative_system = 'Linux'
    if 'Windows' not in operative_system:
        main()
    else:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if world_size == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        master_addr = 'localhost'
        master_port = '12345'  
        world_size = 3
        mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)
# torchrun --nproc_per_node=3 test_pippy.py