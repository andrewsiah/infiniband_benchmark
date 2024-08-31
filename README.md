Infiniband (200Gb/s) connections to GPU06-07. The two machines can use the following addresses to communicate over the IB 200Gb/s link:
GPU06: 10.252.6.196
GPU07: 10.252.6.197

(Please note that the 10Gb Ethernet is avaialble at the 10.252.2.{196,197} addresses, respectively, if you would like to run a comparative test (and let us know how it went, please))


I want to run some benchmarks to compare the performance of  Infiniband and Ethernet.

The Infiniband is a nvidia-provided 200Gb/s link, while the Ethernet is a 100Gb/s link.

Test it using pytorch by transferring data between the two machines.# infiniband_benchmark
