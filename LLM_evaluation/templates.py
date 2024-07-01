

template_general_instruction = """You are an expert microbiologist who given an excerpt from a research paper can easily 
identify the type of relation between a microbe and a disease. Doesn't create new information, but is completely faithful to the information provided, and always gives concise answers."""

template_instruction = """Given the following meaning of the labels, answer the following question with the appropiate label.
positive: This type is used to annotate microbe-disease entity pairs with positive correlation, such as microbe will cause or aggravate the disease, the microbe will increase when disease occurs.
negative: This type is used to annotate microbe-disease entity pairs that have a negative correlation, such as microbe can be a treatment for a disease, or microbe will decrease when disease occurs. 
na: This type is used when the relation between a microbe and a disease is not clear from the context or there is no relation. In other words, use this label if the relation is not positive and not negative."""

template_evidence = """Based on the above description, evidence is as follows: 
{evidence}

"What is the relationship between {microbe} and {disease}?
"""

template_system = template_general_instruction + '\n' + template_instruction
template_user = template_evidence


#template_system = ''
#template_user = template_general_instruction + '\n' + template_instruction + '\n' + template_evidence

