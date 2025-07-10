from langchain_core.prompts import ChatPromptTemplate

rewrite_system = """You are a question re-writer that converts an input question to a better version that is optimized
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", rewrite_system),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")
])