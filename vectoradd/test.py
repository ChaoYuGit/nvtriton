import triton
import triton.language as tl 
import torch

@triton.jit
def add_kernel(a, b, c, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    pos = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = pos < n
    ain = tl.load(a + pos, mask=mask)
    bin = tl.load(b + pos, mask=mask)
    sum = ain + bin 
    tl.store(c + pos, sum, mask=mask)

def add(x, y):
    n = x.numel()
    xy = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x,y,xy,n, BLOCK_SIZE=1024)
    return xy

def main():
    size  = 20480
    x = torch.rand(size, device="cuda:0")
    y = torch.rand(size, device="cuda:0")

    out_torch = x + y
    out_triton = add(x, y)
    print(x)
    print(y)
    print(out_torch)
    print(out_triton)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device="cuda:0", dtype=torch.float32)
    y = torch.rand(size, device="cuda:0", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)

#main()