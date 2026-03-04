
from langchain_core.prompts import PromptTemplate
from model import LLmModel, LangChainLLMWrapper
from vectore_store import VectoreStore
import model_config as conf

def generate_exam_response(retriver, llm, topics=None, num_questions=5, level="easy"):
    prompt = PromptTemplate(
        template=conf.user_template, 
        input_variables=["num_questions", "context", "level"]
    )

    if topics:
        search_query = ", ".join(topics)
    else:
        search_query = "main concept 'not important actually'"
    
    retrived_docs = retriver.invoke(search_query)
    context_text = "\n\n".join([doc.page_content for doc in retrived_docs])
    formatted_user_request = prompt.format(num_questions=num_questions, context=context_text, level=level)
    final_prompt = conf.system_prompt + "\n\n" + formatted_user_request
    response = llm.invoke(final_prompt)

    # context is generated automatically from ConversationalRetrievalChain, so we only need to pass the question
    # response = agent_chain.invoke({"question": formatted_query})
    return response

def main():
    choice = input("Do you want to generate the exam from specific pages or specific topics? (P for Pages, T for Topics):\n")
    while choice.strip().upper() not in ["P", "T"]:
        choice = input("Please enter 'P' for Pages or 'T' for Topics:\n")
    if choice.strip().upper() == "P":
        pages = input("Enter the page numbers separated by ',':\n")
        content = [int(p.strip()) - 1 for p in pages.strip().split(',')]
        type = "pages"
    else:
        topics = input("Enter your topics separated by ',':\n")
        content = topics.strip().split(',')
        type = "topics"
   
    print("Loading PDF and preparing data...")
    
    base_model = LLmModel()
    llm = LangChainLLMWrapper(base_model)

    print("Creating retriever and LLM...")
   
    if type == "pages":
        ret = VectoreStore(file_path="Stages_of_data_mining.pdf", filter_value=content, filter_type="pages")
        
    else:
        ret = VectoreStore(file_path="Stages_of_data_mining.pdf", filter_value=content, filter_type="topics")
        # retriver = MultiQueryRetriever.from_llm(
        #     llm=llm,
        #     retriever=ret(),
        # )

    retriver = ret()
   
    print("Generating exam response...")
    if type == "pages":
        response = generate_exam_response(retriver, llm, num_questions=5, level="easy")
    else:
        response = generate_exam_response(retriver, llm, topics=content, num_questions=5, level="easy")
    
    print("response is:\n\n",response)

if __name__ == "__main__":
    main()