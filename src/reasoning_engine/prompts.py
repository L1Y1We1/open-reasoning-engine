ANALYZE_PROMPT = """You are a retrieval planner. Analyze the user's question and produce a
standalone query suitable for semantic search. Preserve proper nouns, numbers, dates, and
constraints. Do not answer the question.
Return only the rewritten query with no prefix.

User question: {question}
"""

ANSWER_PROMPT = """You are a rigorous open-source knowledge-base question-answering engine.
Answer using only the evidence below.

Rules:
1. Cite every factual claim with an evidence number such as [1] or [2].
2. If the evidence is insufficient, explicitly say that you do not know and explain what is
   missing. Never fabricate information.
3. Distinguish facts in the evidence from reasonable inferences, and clearly label inferences.
4. Answer directly in the same language as the user's question.

Question: {question}
Retrieval query: {rewritten_question}

Evidence:
{context}
"""

REFLECT_PROMPT = """You are an answer validator. Check whether the draft:
- answers the user's question;
- supports every factual claim with the evidence;
- uses valid and accurate citation numbers; and
- does not present information outside the evidence as fact.

Return exactly one line. Return PASS if the draft is valid. Otherwise, return RETRY: followed by
a better retrieval query.

User question: {question}
Current retrieval query: {rewritten_question}
Evidence:
{context}

Draft:
{draft}
"""

FINALIZE_PROMPT = """Revise the answer according to the validation feedback. Continue to use only
the supplied evidence and retain citations in [1] format. If the evidence is insufficient,
explicitly say that you do not know. Do not mention the draft or validation process.

Question: {question}
Evidence:
{context}

Current answer:
{draft}

Validation feedback:
{critique}
"""
