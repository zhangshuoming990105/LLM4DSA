from models import Chat, num_tokens_from_messages
from prompts import *
import os
chatbot = Chat(model="gpt-4o")
chatbot.system_prompt = """You are an expert performance engineer with experience in optimizing C code.
Your job is to translate an arbitrary C code with automatic vectorization, matrix multiplication, and parallelization optimizations.
You need to do this job with careful thinking first and optimize the code step by step."""

chatbot.chat(analyzer_prompt_step1)

