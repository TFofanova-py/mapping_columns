from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chains.sequential import SequentialChain
from langchain.prompts import PromptTemplate
import json

llm = OpenAI(openai_api_key=json.load(open("openai_creds.json"))["OPENAI_API_KEY"],
             temperature=0)

candidates_template = """I need assistance in mapping data from a source to template columns. For each column from 
the template, select candidates corresponding columns from the source.Template numeric fields should only match 
columns with numeric data of similar order.Date data - columns containing dates If the data contains anything other 
than numbers, it is not considered numeric

Please provide me with a json, contains for every column in template table detailed mapping strategy for aligning the 
column with the corresponding fields in the source. The output must be in format: 
filed_name: [source_column_name],
...

Output must contain only json.

Here are the details:
**Source Table**:
- Columns:
{source_without_column_names}

**Template Table**:
- Template Fields:
{template}
"""
candidates_prompt_template = PromptTemplate(input_variables=["source_without_column_names", "template"],
                                            template=candidates_template)

candidates_chain = LLMChain(llm=llm, prompt=candidates_prompt_template, output_key="candidates")

associate_template = """
For each field in Output modify corresponding list by replacing each value in the list using Dictionary.

The output must be in format:
filed_name: [source_column_names],
...

Output must contain only json.


Thank you. Now, for each field in candidates, select one most relevant candidate, based on 
examples, data length, column name from source table and template.

**Output**:
{candidates}

**Dictionary**:
{columns_dict}

**Template Table**:
- Template Fields:
{template}

The output must be in format:
filed_name: source_column_name,
...
"""

associate_prompt_template = PromptTemplate(input_variables=["candidates", "columns_dict", "template"],
                                           template=associate_template)
associate_chain = LLMChain(llm=llm, prompt=associate_prompt_template, output_key="associates")

mapping_template = """Given output matches a field in template table with a column in source table. 

For each field in Output create Python lambda function to transform source column. 
The lambda functions must assume 
that the datetime columns in the source table are of type `str`, numeric columns are of type 'float'.

The output must be in format:
filed_name: source_column_names, transformation
...

Output must contain only json.

**Output**:
{associates}

**Source Table**:
- Columns:
{source}

**Template Table**:
- Template Fields:
{template}
"""

mapping_prompt_template = PromptTemplate(input_variables=["associates", "source", "template"],
                                         template=mapping_template)
mapping_chain = LLMChain(llm=llm, prompt=mapping_prompt_template, output_key="mapping")

overall_chain = SequentialChain(chains=[candidates_chain, associate_chain, mapping_chain],
                                input_variables=["source_without_column_names", "columns_dict", "source", "template"],
                                output_variables=["mapping"],
                                verbose=True
                                )
