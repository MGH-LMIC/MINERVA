

template_general_instruction = """You are an expert microbiologist who given an excerpt from a research paper can easily 
identify the type of relation between a microbe and a disease. Doesn't create new information, but is completely faithful to the information provided, and always gives concise answers."""

template_specific_instruction = """There are only four possible types of relations between a microbe and a disease, please read their meaning carefully: 

a) POSITIVE: Choose this alternative if ANY of the following statements are TRUE or IMPLIED (Causality is NOT REQUIERED):
    - Positive correlation between the presence or abundance of the microbe and the presence or gravity of the disease
    - Microbe will cause or aggravate the disease
    - Abundance or presence of the microbe will increase when the disease occurs
    - Absence of the disease correlates with decreased abundance of the microbe
    - Presence of the disease correlates with increased abundance of the microbe
b) NEGATIVE: Choose this alternative if If ANY of the following statements are TRUE or IMPLIED (Causality is NOT REQUIERED):
    - Negative correlation between the presence of the microbe and the presence or gravity of the disease
    - Microbe can be a treatment, beneficial to cure, or protects against the disease
    - Abundance or presence of the microbe will decrease when the disease occurs
    - Presence of the disease correlates with decreased abundance of the microbe
    - Absence of the disease correlates with increased abundance of the microbe
c) RELATE: Choose this alternative if there is a relation between the disease and the presence or absence of microbe BUT the nature of the relation is not clear from the information in the excerpt
d) NA: Choose this alternative if Microbe and disease appear in the excerpt but there is no link or relation between them

You can only select ONE ALTERNATIVE for each excerpt.
"""

template_excerpt = """Here is an excerpt for you to analyze: 
{sentence}"""

template_question = """Based only in the excerpt how is disease '{disease}' and microbe '{microbe}' related? 
Lets think step by step:
"""

template_question_train = """Based only in the excerpt, and the definitions above the ground truth relation between disease {disease} and microbe {microbe} is {relation}. 
- Assume {relation} is the correct relation between {disease} and {microbe} and that no more information is needed.
- Please provide a step by step reasoning process to reach to this conclusion. 
- In case the relation is negative or positive mention which of the conditions mentioned above are TRUE.
- Explicitly follow the following two steps in your reasoning process:

Step 1: Explain in no more than 20 words why {relation} is the correct alternative highlighting which word or phrase is key for finding the correct answer.

Step 2: Always conclude with the following statement and NOTHING ELSE: Therefore the correct alternative is {relation}

Your turn, please follow this scheme of the two steps in your reasoning (and explicitly mention Step 1 and Step 2) to explain why '{relation}' is the correct relation, because in fact it is. 
Lets think step by step"""

# template_question_train = """Based only in the excerpt above and the definition of the four types of relations, what is the correct relation between disease '{disease}' and microbe '{microbe}'?
# Lets think step by step and follow a three step explanation where in:
# Step 1: Provide your explanation of the incorrect alternatives
# Step 2: Provide your explanation of the the correct alternative and why
# Step 3: Finalize with the followin phrase: 'Therefore, the correct alternative is .... (mention there the correct alternative)'
# Lets think step by step"""



template_question_train = """Based only in the excerpt, the correct relation between disease '{disease}' and microbe '{microbe}' is '{relation}'. 
- Please provide a step by step reasoning process to reach to this conclusion. 
- Use the evidence in the excerpt to establish which of the conditions above are fulfilled as your explanation.
- Follow the following two steps in your reasoning process:

STEP1: Explain in no more than 20 words why '{relation}' is the correct alternative, highlight which word or phrase is key for reaching to this conclusion.
- {relation}: Provide your explanation in no more than 20 words

STEP2: Conclusion.
Finalize with this statement and NOTHING ELSE: 'Therefore the correct alternative is {relation}'

Your turn, please follow this scheme of the two steps in your reasoning process but don't repeat the instructions in your answer. 
Lets think step by step"""