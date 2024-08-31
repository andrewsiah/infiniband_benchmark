Code to benchmark the performance of Infiniband and Ethernet.




For ethernet.

To use:
On GPU07:
```
python benchmark.py --backend nccl --rank 1 --world-size 2 --master-addr 10.252.2.196 --master-port 29501 
```


On GPU06:
```
python benchmark.py --backend nccl --rank 0 --world-size 2 --master-addr 10.252.2.196 --master-port 29501 
```