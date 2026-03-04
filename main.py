from langchain_classic.chains import ConversationalRetrievalChain 
from langchain_classic.memory import ConversationBufferMemory 
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from model import LLmModel, LangChainLLMWrapper
from vectore_store import VectoreStore
import model_config as conf

def generate_exam_response(retriver, llm, num_questions=5, question_formate="draw circle (MCQ)", level="easy"):
    prompt_template = conf.system_prompt
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriver,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    formatted_query = f"Generate an exam with {str(num_questions)} questions, in the formate of {str(question_formate)}, and the difficulty level is {level}."

    # context is generated automatically from ConversationalRetrievalChain, so we only need to pass the question
    response = agent_chain.invoke({"question": formatted_query})
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
        retriver = ret()
    else:
        ret = VectoreStore(file_path="Stages_of_data_mining.pdf", filter_value=content, filter_type="topics")
        retriver = MultiQueryRetriever.from_llm(
            llm=llm,
            retriever=ret(),
        )
   
    print("Generating exam response...")
    
    response = generate_exam_response(retriver, llm, num_questions=5, question_formate="draw circle (MCQ)", level="easy")
    
    print("response is:\n\n",response["answer"])

if __name__ == "__main__":
    main()