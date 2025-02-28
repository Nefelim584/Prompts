import os
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from together import Together
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableSerializable
from pydantic import Field, PrivateAttr
from openai import OpenAI
from giskard.llm.client.openai import OpenAIClient
from dotenv import load_dotenv
from giskard import Dataset, Model, scan
import giskard.llm
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

_client = Together()  

oc = OpenAIClient(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", client=_client)


giskard.llm.set_default_client(oc)

pd.set_option("display.max_colwidth", None)

IPCC_REPORT_URL = "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf"
TEXT_COLUMN_NAME = "query"

PROMPT_TEMPLATE = """You are the Climate Assistant, a helpful AI assistant made by Giskard.
Your task is to answer common questions on climate change.
You will be given a question and relevant excerpts from the IPCC Climate Change Synthesis Report (2023).
Please provide short and clear answers based on the provided context. Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""


def get_context_storage() -> FAISS:
    """Initialize a vector storage of embedded IPCC report chunks (context)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    docs = PyPDFLoader(IPCC_REPORT_URL).load_and_split(text_splitter)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db


prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])


class TogetherLlamaModel(RunnableSerializable):
    model_name: str = Field(default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    temperature: float = Field(default=0.7)
    _client: Together = PrivateAttr()  

    def __init__(self, **data):
        super().__init__(**data)
        self._client = Together()  

    def invoke(self, input_text: str, config=None, **kwargs) -> str:
        kwargs.pop("stop", None)
        prompt_text = str(input_text)
        messages = [{"role": "user", "content": prompt_text}]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return response.choices[0].message.content



llm = TogetherLlamaModel()


climate_qa_chain = RetrievalQA.from_llm(
    llm=llm,
    retriever=get_context_storage().as_retriever(),
    prompt=prompt
)


print(climate_qa_chain.invoke("Is sea level rise avoidable? When will it stop?"))



class FAISSRAGModel(Model):
    def model_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[TEXT_COLUMN_NAME].apply(lambda x: self.model.run({"query": x}))


giskard_model = FAISSRAGModel(
    model=climate_qa_chain,
    model_type="text_generation",  # Options: regression, classification, or text_generation.
    name="Climate Change Question Answering",
    description="This model answers any question about climate change based on IPCC reports",
    feature_names=[TEXT_COLUMN_NAME]
)


giskard_dataset = Dataset(pd.DataFrame({
    TEXT_COLUMN_NAME: [
        "According to the IPCC report, what are key risks in Europe?",
        "Is sea level rise avoidable? When will it stop?"
    ]
}), target=None)


print(giskard_model.predict(giskard_dataset).prediction)

full_results = scan(giskard_model, giskard_dataset)
print(full_results)