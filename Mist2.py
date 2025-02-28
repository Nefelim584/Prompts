import os
import torch
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from giskard import Dataset, Model, scan
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
login(HUGGINGFACE_API_KEY)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
quant_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=quant_config, device_map="auto"
)

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

RESULTS_FILE = "results_mist2.txt"
ERROR_LOG_FILE = "error_log_mist2.txt"

def llm_response(df: pd.DataFrame) -> list:
    responses = []
    with open(RESULTS_FILE, "w", encoding="utf-8") as results_file, open(ERROR_LOG_FILE, "w", encoding="utf-8") as error_log:
        for prompt in df[TEXT_COLUMN_NAME]:
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=512)
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response_text)
                results_file.write(f"Prompt: {prompt}\nResponse: {response_text}\n{'-'*50}\n")
                

            except Exception as e:
                error_log.write(f"Prompt: {prompt}\nError: {str(e)}\n{'-'*50}\n")
                responses.append(f"Error generating response: {str(e)}")

    return responses

giskard_model = Model(
    model=llm_response,
    model_type="text_generation",
    name="Mistral 7B Climate QA",
    description="A climate question answering model using Mistral 7B and IPCC reports",
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
