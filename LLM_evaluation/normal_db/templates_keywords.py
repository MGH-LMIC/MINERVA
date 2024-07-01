

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

template_get_keywords = """From the excerpt, What is the keywords or short phrase that highlights how disease {disease} and microbe {microbe} are related? 
Please consider the following instructions as you choose the Keywords or phrases from the excerpt:
- Your selection should be as concise as posible and just write the chosen word or phrase from the excerpt and NOTHING ELSE.
- DO NOT provide an explanation 
- DO NOT use {microbe} or {disease} as a Keywords.
- DO NOT use any microbe or bacteria nor any disease or condition as Keywords, just words that point out the relation between {disease} and {microbe}"""


#template_get_keywords = """Based on the excerpt, extract the Keywords or phrases that better express how disease {disease} is related with {microbe}.
#- Your selection should be as concise as posible and just write the chosen word or phrase from the excerpt and NOTHING ELSE.
#- DO NOT use {microbe} or {disease} as a KEYWORD. """