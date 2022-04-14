## Introduction

At a high level, instead of having one efficient CPU that handles 1 thread (or 2 using hyperthreading)
per CPU core, a GPU has much more threads that are slower than a CPU. A typical CPU has 4 cores which translates
to 8 virtual cores (as in the 4 cores can handle 8 threads. On the other hand, a GPU can handle millions of threads.

### Efficient jobs on a GPU

Efficient GPU code is sufficiently parallelized and then relies on merging the work of thousands or millions of
threads back into the value a user cares about. Jobs are allocated to **threads**, each thread handles one
simple instance of execution. Threads are grouped into

### Anatomy of a GPU

A GPU has multiple levels to its structure hierarchy.

#### Stream Multiprocessors

Have limited number of registers (~256KB)

occupancy threads and few registers on board


### Memory Structure

Ignoring all the memories dedicated for graphics, there are

- SM Registers, usually have ~256KB of memory per SM. One thread can use the register.
- Every SM has an L1 Cache/SMEM (Shared Memory) (~192KB). Shared Memory is on a single SM, all threads launched on a SM can share the SMEM. All blocks running on a SM can share the SM too.
- SM Read-Only Memory, stores instructions, constants and textures. This is read-only to the kernel.
- L2 Cache shared by all SMs. Usually ~40MB
- Global Memory. Framebuffer size of the GPU and DRAM sitting in the GPU

#### Why 32-bit numbers

1. the SMP registers are 32-bit and are usually the largest constraint for the size of a block, a 64-bit value would need twice the space and therefore the number of occupancy threads will go down when we are optimally running kernels.

2. GPUs seem to run 1/16 times slower when they are operating on 64-bit values.
