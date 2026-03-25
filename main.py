# import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()


def main():
    print("Hello from langchain-course!")
    
    information = """
    Bengaluru,[b] also known as Bangalore (its official name until 1 November 2014), is the capital and largest city of the southern Indian state of Karnataka. As per the 2011 census, the city had a population of 8.4 million, making it the third most populous city in India and the most populous in South India. The Bengaluru metropolitan area had a population of around 8.5 million, making it the fifth most populous urban agglomeration in the country. It is located towards the southern end of the Deccan Plateau, at an altitude of 900 m (3,000 ft) above sea level. The city is known as India's "Garden City", due to its parks and greenery.

Archaeological artefacts indicate that the human settlement in the region happened as early as 4000 BCE. The first mention of the name "Bengaluru" is from an old Kannada stone inscription from 890 CE found at the Nageshwara Temple in Begur. The region was ruled by the Western Ganga dynasty since 350 CE, and became part of the Chola Empire in the early eleventh century CE. In the late Middle Ages, it formed a part of the Hoysala kingdom and then the Vijayanagara Empire. In 1537 CE, Kempe Gowda I, a feudal ruler under the Vijayanagara Empire, established a mud fort which is considered the foundation of the modern city of Bengaluru with the earlier established areas (petes) still in existence. After the fall of the Vijayanagara Empire, Kempe Gowda declared independence, and the city was expanded by his successors. In 1638 CE, an Adil Shahi army defeated Kempe Gowda III, and the city became a jagir (feudal estate) of Shahaji. The Mughals later captured the city and sold it to Chikka Devaraja Wodeyar, the Maharaja of the Kingdom of Mysore. After the death of Krishnaraja Wodeyar II in 1759 CE, Hyder Ali seized control of the kingdom and with it, the administration of Bengaluru, which passed subsequently to his son, Tipu Sultan.

The city was captured by the British East India Company during the Anglo-Mysore Wars, and became part of the princely state of Mysore. The administrative control of the city was returned to Krishnaraja Wodeyar III, then Maharaja of Mysore, and the old city re-developed under the dominion of the Mysore kingdom. In 1809 CE, the British formed a military garrison in the city and established the cantonment outside the old city. In the late 19th century CE, the city was composed of two distinct urban settlements, the old Pete and the new Cantonment. Following India's independence in 1947, Bengaluru became the capital of Mysore State, and remained the capital when the state was enlarged and unified in 1956 and subsequently renamed as Karnataka in 1973. The two urban settlements which had developed as independent entities, merged under a single urban administration in 1949.

Bengaluru is one of the fastest-growing metropolises in India. As of 2023, the metropolitan area had an estimated GDP of $359.9 billion, and is one of the most productive metro areas of India. The city is a major centre for information technology (IT), and is consistently ranked amongst the world's fastest growing technology hubs. As the largest hub and exporter of IT services in the country, it is regarded as the "Silicon Valley of India". Manufacturing is a major contributor to the economy with the city home to several state-owned units. The city also hosts several prominent institutes of higher education and of national importance.
    """

    summary_template = """
    given the information {information} about a city I want you to create:
    1. A short summary with three bullet points
    2. Two interesting facts about the city that kids may like it.
    """

    summary_prompt_template = ChatPromptTemplate.from_template(summary_template).partial(information=information)

    llm = ChatOllama(temperature=0, model="gemma3:12b")
    # chain = summary_prompt_template | llm | StringOutputParser().with_config(config={"transformations": [
    #     "trim",
    #     "capitalize",
    # ]})

    chain = summary_prompt_template | llm

    response = chain.invoke({})
    console = Console()
    console.print(Markdown(response.content))

if __name__ == "__main__":
    main()
