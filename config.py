import os
PPLX_API_KEY = "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"
QWEN_API_KEY="sk-5f89fd2525f141aaa14354c854a5137c"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY="sk-de942e3222de415cb5636482c0b6a378"
PPLX_AVAILABLE_MODELS = [
    "mixtral-8x7b-instruct",
    "llama-3.1-sonar-large-128k-chat",
    "llama-3.1-sonar-huge-128k-online",
    "llama-3.1-70b-instruct",
]

GPT_AVAILABLE_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
]

ANTHROPIC_AVAILABLE_MODELS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

DEEPSEEK_AVAILABLE_MODELS = [
    "deepseek-coder",
]

QWEN_AVAILABLE_MODELS = [
    "qwen-max-0919",
    "qwen-max",
    "qwen-coder-turbo-latest",
    "qwen-long",
]

OLLAMA_AVAILABLE_MODELS = [
    # general models
    "llama3.1",
    # code models
    "codegeex4",
    "codegeex4-9b-all-fp16",
    "codestral",
    "codestral:22b-v0.1-f16",
    "deepseek-coder-v2",
    "deepseek-coder-v2:16b-lite-instruct-fp16",
    "codegemma",
    "codeqwen",
    
]