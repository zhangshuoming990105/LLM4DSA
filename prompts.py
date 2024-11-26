"""This file contains the prompts for the AI model to translate 
domain-specific applications into code with domain-specific operation supports without modification to compilers.
We currently target C code.
Each step should be accompanied with a 1-shot example to demonstrate the expected input and output.
"""

translator_system_prompt = """You are an expert performance engineer with experience in optimizing C code.
Your job is to translate an arbitrary C code with automatic vectorization, matrix multiplication, and parallelization optimizations.
You need to do this job with careful thinking first and optimize the code step by step.
All the code snippets you inpur or output should be marked between '```c' and '```' for alignment and formatting.
If you need to generate multiple output, return them in a list, I mean, return with '[' and ']', and separate them with ','."""

"""
input: C code snippet
(Natural language -> repository -> CUDA -> C) -> aladdin-C
Tree 1:
input->(analyze if valid C code)->(analyze current functions)->(analyze single function)->simple function
                                    ->multiple functions->inputs         ->extract functions->inputs
                                        + function relationships                + function relationships
Tree 2:
simple function->(analyze)->(transform)->(unittest)->optimized function
Tree 3:
multiple optimized functions->(combine)->(end2end test)->optimized code snippet                                  
output: optimized code snippet
"""

analyzer_prompt_step1 = """#Step 1:
based on the input code snippet, you need to analyze whether the input is a valid C code, with all the necessary libraries and functions provided.
Note that the input code snippet should be at least a complete function or a set of functions that can be compiled and run.
You should return with 'YES' to indicate it is valid, or 'NO' to indicate it is invalid. You can also provide additional information if needed."""
analyzer_prompt_step2 = """#Step 2:
based on the input C code snippet, you need to analyze the functions, global values and their relationships.
If it contains only one function, return with the original input code snippet,
otherwise, you need to split the functions"""

# Prompt for analyzing a single function
analyzer_prompt_single_function = """#Single Function Analysis:
Analyze the given single function to understand its purpose, inputs, outputs, and any dependencies it might have.
Provide a detailed breakdown of the function's logic and any potential areas for optimization."""

# Prompt for analyzing multiple functions
analyzer_prompt_multiple_functions = """#Multiple Functions Analysis:
Analyze the given set of functions to understand their individual purposes, inputs, outputs, and interdependencies.
Provide a detailed breakdown of each function's logic, how they interact with each other, and any potential areas for optimization."""

# Prompt for extracting functions and their relationships
extractor_prompt_functions_relationships = """#Extract Functions and Relationships:
Identify and extract all functions from the input code snippet, along with their relationships.
Provide a clear mapping of function calls, shared variables, and any other dependencies."""

# Prompt for transforming a simple function
transformer_prompt_simple_function = """#Transform Simple Function:
Transform the analyzed simple function to optimize its performance.
Consider techniques such as loop unrolling, inlining, and other micro-optimizations.
Ensure the transformed function maintains the same functionality."""

# Prompt for unit testing an optimized function
unittest_prompt_optimized_function = """#Unit Test Optimized Function:
Create unit tests for the optimized function to ensure it behaves as expected.
Include tests for edge cases and typical usage scenarios."""

# Prompt for combining multiple optimized functions
combiner_prompt_multiple_optimized_functions = """#Combine Multiple Optimized Functions:
Combine the optimized functions into a cohesive code snippet.
Ensure that the combined code maintains the original functionality and is optimized for performance."""

# Prompt for end-to-end testing of the optimized code snippet
end2end_test_prompt_optimized_code = """#End-to-End Test Optimized Code Snippet:
Perform an end-to-end test on the combined optimized code snippet.
Verify that the code snippet compiles, runs correctly, and meets performance expectations."""
