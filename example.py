from models import Chat, num_tokens_from_messages

chatbot = Chat(model="gpt-4o-mini")
chatbot.chat("What is the capital of France?")
print(chatbot.response())
print(num_tokens_from_messages(chatbot.messages))

# ask follow-up questions
chatbot.chat("What is the capital of UK? compare these two capital.")
print(chatbot.response())
print(num_tokens_from_messages(chatbot.messages))