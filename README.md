# LLM for agile DSA support in compilers

## Motivations:

- there are many DSAs, but there are only a few backends in compilers like LLVM, even develop a DSA-related compiler code requires several months of development.
- to support a DSA, no need to develop a new backend, we only need to support three things: how to identify patterns could use DSA; how to correctly translate identified patterns with DSA instructions; how to measure cost with different DSA options.
- LLMs can help greatly in this story. First, from a language-manual level specification(a pdf), LLMs can output structured instruction specifications(a json), and register programming models. Second, LLMs can generate both rules and examples of expected translation. Third, LLMs can automatically synthesize unittests for code correctness evaluation.

## Challenges and methods:

1. parse language manual: from raw language manual, can we extract **structured outputs** from it? We achieve this by using LLMs with xxxx.
2. find potential DSA workloads: a DSA is usually designed for specific usecases, therefore, generalizability is not a concern. So current problem is: for a specific program P, can we find all potentially DSA-runnable code patterns? We observe that LLMs are good at finding potential workloads, they act like auto-vectorizer, but to more cases beyond vectorization.
3. translate existing code to DSA instructions: 
4. evaluate the cost of DSA-adaptation: 


