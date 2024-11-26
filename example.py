from models import Chat, num_tokens_from_messages

chatbot = Chat(model="gpt-4o")
chatbot.system_prompt = "You are an expert performance engineer with experience in optimizing numerical linear algebra kernels."
input_prompt = """I need help with optimizing a numerical kernel. It is written in a Python DSL
for code optimization called Exo, which is similar to Halide.
Here are my relevant hardware details:
- The target hardware is an x86 CPU with AVX2 support.
- We will be targeting single-core execution, so you can ignore parallelism.
- L1 instruction cache size: 32 KB
- L1 data cache size: 48 KB
- L2 cache size: 2 MB
- L3 cache size: 36 MB
Here is the kernel I need help with, written in Exo:
def doitgen(A: f32[64, 64, 64] @ DRAM, C4: f32[64, 64] @ DRAM,
sum: f32[64] @ DRAM):
for r in seq(0, 64):
for q in seq(0, 64):
for p in seq(0, 64):
sum[p] = 0.0
for s in seq(0, 64):
sum[p] += A[r, q, s] * C4[s, p]
for p in seq(0, 64):
A[r, q, p] = sum[p]
Currently I get 5.64 GFLOPS.
Please provide a step-by-step plan for optimizing the kernel.
Once you have a plan, begin optimizing the kernel by giving me a series of
commands, each of which are described below. I will apply the command one at a
time, and provide you with the new kernel code and its performance.
You can use the following commands:
```json
{"optimization": "tile", "description": "tile the loop at 'line' with 'tile_size'", "arguments": {"line": "source code of line to apply
tiling", "tile_size": "size of the tile", "outer_name": "name of the new outer loop variable", "inner_name": "name of the new inner
loop variable"}}
{"optimization": "fuse", "description": "fuse the loops at 'line1' and 'line2'", "arguments": {"line1": "source code of first line to
fuse", "line2": "source code of second line to fuse"}}
{"optimization": "reorder", "description": "reorder the outer loop at 'line' to be the inner loop", "arguments": {"line": "source code
of outer loop to reorder"}}
{"optimization": "fission", "description": "fission the loop containing 'line' into two consecutive loops, splitting at 'line'", "
arguments": {"line": "source code of line to fission at", "location": "split (before | after) 'line'"}}
{"optimization": "unroll", "description": "fully unroll the loop at 'line'", "arguments": {"line": "source code of loop to unroll"}}
```
When you are ready to give the optimization, end your message with JSON as follows:
```
APPLY: {"optimization": "optimization name", "arguments": {"arg1": "value1", ...}}
```
Only give a single command at a time. Do not give multiple commands in a single
message.
Important Notes:
- Consider reordering loops before applying tiling or vectorization.
- Reorder loops to maximize data reuse and minimize cache misses.
- Assume matrices are stored in row-major order.
- If the same source line appears multiple times in the code, end the line
with ' #N' where 'N' is the occurrence number (0-indexed). For example,
'for i in seq(0, 32): #0'."""
chatbot.chat(input_prompt)
print(chatbot.response())
response = """To optimize the given kernel, we will focus on improving data locality, minimizing cache misses, and leveraging vectorization. Here is a step-by-step plan:

1. **Reorder Loops**: Reorder the loops to maximize data reuse and minimize cache misses. Since matrices are stored in row-major order, we should try to access data in a contiguous manner.

2. **Tile Loops**: Apply loop tiling to improve cache utilization by fitting working data sets into the cache.

3. **Unroll Loops**: Fully unroll loops where possible to reduce loop overhead and increase instruction-level parallelism.

4. **Vectorization**: Ensure that the innermost loop is vectorized to take advantage of AVX2 instructions.

Let's start with the first step: reordering the loops to improve data locality. We will reorder the `p` loop to be the innermost loop in the computation of `sum[p]`, as this will allow us to access `A[r, q, s]` and `C4[s, p]` in a more cache-friendly manner.

### Step 1: Reorder Loops

We will reorder the loops so that the `p` loop is the innermost loop in the computation of `sum[p]`.

```json
APPLY: {"optimization": "reorder", "arguments": {"line": "for p in seq(0, 64): #0"}}
```"""






# chatbot = Chat(model="gpt-4o")
# chatbot.chat("What is the capital of France?")
# print(chatbot.response())
# print(num_tokens_from_messages(chatbot.messages))

# # ask follow-up questions
# chatbot.chat("What is the capital of UK? compare these two capital.")
# print(chatbot.response())
# print(num_tokens_from_messages(chatbot.messages))