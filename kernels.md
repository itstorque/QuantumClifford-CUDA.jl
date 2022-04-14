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

## Atomic Execution

```julia
function xor_atomic(a, b)
    return
end
```

```julia
function atomic_singlethread(l::AbstractArray{T}, r::AbstractArray{T}, out) where {T}
    return
end
```
