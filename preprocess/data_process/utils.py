def make_prefix(dp, template_type):
    question = dp['question']
    if template_type == 'autorefine':
        prefix = f"""You are a helpful assistant excel at answering questions with multi-turn search engine calling. \
To answer questions, you must first reason through the available information using <think> and </think>. \
If you identify missing knowledge, you may issue a search request by outputting <search> query </search>. \
The system will return relevant documents enclosed in <documents> and </documents>. \
The system may also provide extracted factual nuggets enclosed in <facts> and </facts> (inside the documents block). \
After each search, you must update a compact working memory in <memory> and </memory>. \
Memory should be concise, deduplicated, and grounded (cite Doc IDs when possible). \
You may send multiple search requests if needed. \
Once you have sufficient information, provide a concise final answer enclosed in <answer> and </answer>. \
IMPORTANT: The <answer>...</answer> block MUST be the last block in your output. \
For example, <answer> Donald Trump </answer>. Question: {question}\n"""
    elif template_type == 'searchr1':
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can output <search> query </search> to search. \
The search engine will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can output <answer> answer </answer>. \
For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix
