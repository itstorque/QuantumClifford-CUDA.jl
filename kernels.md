# Writing Efficient GPU Kernels in CUDA.jl

This will cover writing kernels, see [/syntax.md](syntax.md)
for details on importing and writing functions. For these
examples, I will implement different xor reduction functions
and commutation kernels to compare the different speed-ups
offered.

The xor reductions `xor_*` will reduce an array of `UInt32`s
of length `N` to a single `UInt32` output value `a[N] -> b[1]`.
The function applies xor's all the values together.

The commutation `comm_*` functions will take in two arrays
of `UInt32`s of length `N` to a single `UInt32` output value
`l[N], r[N]-> out[1]`. Here, `l, r` are two arrays storing
`X, Z` information, i.e. `l = [l_X, l_Z]`. The commutator
is defined as `(l_Z & r_X) ⊻ (l_X & r_Z) mod 4`, since
the commutation of two operators can take one of 4 values:
`+1`, `-1`, `+i`, `-i`. For these examples I will skip
over the `mod 4` part for simplicity only.

Runtime will be compared to that of a CPU and a GPU on
arrays of size

## Naïve Implementation

Make one thread on a GPU loop through and xor into one
register.

```julia
function xor_singlethread(a, b)
    for i in 1:length(a)
        b[] = b[] ⊻ a[i]
    end
    return
end
```

The same procedure for the commutator, except now iterate one more time
to count the number of ones at the output.

```julia
function comm_singlethread(l::AbstractArray{T}, r::AbstractArray{T}, out) where {T}

    thread = threadIdx().x

    len = length(l) >> 1

    for i in 1:len+1
        if i <= len        
            # uncomment line below to see how what each thread runs
            # @cuprintln "thread $thread: $i = $(out[]) ⊻ ($(l[i+len] & r[i]) ⊻ $(l[i] & r[i+len]))"
            out[] = out[] ⊻ ((l[i+len] & r[i]) ⊻ (l[i] & r[i+len]))
        else
            @cuprintln "thread $thread: count_ones"
            out[] = count_ones(out[])
        end
    end

    return

end
```


#### Runtime

## Atomic Execution

To get a speed-up, we need to subdivide the tasks along threads. Since the
code runs in instances, each instance comes with a `threadIdx()` and
`blockDim()` which helps identify them. One way of offloading work
so that multiple cores work on it is by making every thread xor a value
of `a` with the output value atomically using the `@atomic` or `CUDA.@atomic` tag.

```julia
function xor_atomic(a, b)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    CUDA.@atomic b[] = b[] ⊻ a[index]
    return
end
```

```julia
function atomic_atomic(l::AbstractArray{T}, r::AbstractArray{T}, out) where {T}
    return
end
```

## Grid Reduction

One side effect of our previous implementation is stalling time.
Threads can't all write together over the atomic assignment
(since that would throw away some threads operations).

One way to tackle this is by reducing the values in a tree-like
fashion, this method is called "grid reduction". In grid reduction,
we allocate threads that operate on `N` items on the first run,
applying an operation on 2 of the `N` items. In this case, we will
xor every adjacent pair of elements. After the first iteration,
we we will have `N/2` items, we can keep repeating that until we
reach `1` item left. The text diagram below showcases a grid
reduction example on 16 values.

```
1 2 3 4 5 6 7 8 9 A B C D E F     (16 items)
|/  |/  |/  |/  |/  |/  |/  |
1 - 3 - 5 - 7 - 9 - B - D - F     ( 8 items)
| /     | /     | /     | /
1 - - - 5 - - - 9 - - - D - -     ( 4 items)
|     /         |    /
1 - - - - - - - 9 - - - - - -     ( 2 items)
|             /
1 - - - - - - - - - - - - - -     ( 1 item )
```

This ends up operating in `O(log(N))` runs.

```julia
function xor_grid(a, b)

    elements = blockDim().x*2
    thread = threadIdx().x

    # serial reduction of values across blocks
    i = thread+elements
    while i <= length(a)
        a[thread] = a[thread] ⊻ a[i]
        i += elements
    end

    # parallel reduction of values in a block
    d = 1
    while d < elements
        sync_threads()
        index = 2 * d * (thread-1) + 1
        @inbounds if index <= elements && index+d <= length(a)
            a[index] = a[index] ⊻ a[index+d]
        end
        d *= 2
    end

    if thread == 1
        b[] = a[1]
    end

    return

end
```

```julia
function atomic_grid(l::AbstractArray{T}, r::AbstractArray{T}, out) where {T}
    return
end
```
