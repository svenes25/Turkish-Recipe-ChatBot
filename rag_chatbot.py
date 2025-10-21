import pandas as pd
from dotenv import load_dotenv
import os
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.agent import AgentExecutor 
from functools import partial
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import torch

print(torch.__version__)
print(torch.version.cuda)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.")

LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)


def load_and_split_documents():
    """Yerel bir dosyadan (CSV) veri setini yükler ve parçalara ayırır."""
    file_path = "datav2.csv"
    try:
        # Sadece ilk 10 satırı yükleme mantığı korunmuştur.
        df = pd.read_csv(file_path, nrows=10)
        print(f"'{file_path}' dosyasının ilk 10 satırı başarıyla yüklendi.")

        print("\n--- Yüklenen Tarif Başlıkları ---\n")
        print(df['Title'])
    except FileNotFoundError:
        print(f"Hata: '{file_path}' dosyası bulunamadı. Lütfen aynı dizinde olduğundan emin olun.")
        return []

    docs = []
    for index, row in df.iterrows():
        try:
            content = f"Yemek Adı: {row['Title']}\n\nMalzemeler: {row['Materials']}\n\nTarif: {row['How-to-do']}"
            metadata = {"title": row['Title']}
            docs.append(Document(page_content=content, metadata=metadata))
        except KeyError as e:
            print(f"Hata: '{e.args[0]}' sütunu veri setinde bulunamadı.")
            return []

    print(f"{len(docs)} adet tarif dökümanı yüklendi.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Toplam {len(splits)} adet metin parçasına bölündü.")
    return splits


def create_vector_store(splits):
    if not splits:
        print("Uyarı: 'splits' listesi boş. Vektör veritabanı oluşturulamadı.")
        return None
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding işlemi için kullanılan cihaz: {device.upper()}")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': device} # Modeli belirtilen cihaza taşı
    )

    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)

    return vector_store

@tool
def retrieve_context(query: str, vector_store: Chroma):
    """Yemek tarifleriyle ilgili bir sorguyu yanıtlamaya yardımcı olacak bilgileri getir."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    
    serialized = "\n\n".join(
        (f"YEMEK ADI: {doc.metadata.get('title', 'N/A')}\nDETAYLAR: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized


def create_recipe_agent(vector_store):
    """
    Vektör deposuna erişmek için retrieve_context aracını kullanan bir Agent oluşturur.
    Önceki RAG zincirlerinin (howto, material, title) yerini alır.
    """
    retrieval_tool_partial = partial(retrieve_context, vector_store=vector_store)
    retrieval_tool = tool(retrieval_tool_partial, name="retrieve_recipe_info", description="Kullanıcının sorgusuyla ilgili yemek tarifleri ve malzemeleri getirir. Yemek adlarını, malzemeleri veya nasıl yapılacağını öğrenmek için kullanın.")

    tools = [retrieval_tool]

    system_prompt = (
        "SEN PROFESYONEL BİR ŞEFSİN. Kullanıcı sorularını yanıtlamak için daima 'retrieve_recipe_info' "
        "adlı aracı kullanmalısın. Gerekirse aracı birden fazla kez çağır. Sadece aracın döndürdüğü "
        "bilgilere dayanarak, detaylı ve kibar bir şekilde cevap ver. Eğer araç bilgi döndürmezse, "
        "\"Üzgünüm, veri setimde bu tarife dair bir bilgi bulunmamaktadır.\" diye cevap ver."
    )
    
    agent = create_agent(LLM, tools, prompt=system_prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor


def ask_agent(agent_executor, question):
    """Oluşturulan AgentExecutor ile soru sorar."""
    print(f"\nSoru: {question}")
    response = agent_executor.invoke({"input": question})
    return response["output"]


if __name__ == "__main__":
    splits = load_and_split_documents()

    if splits:
        print("\n--- İlk 3 Döküman Parçası ---\n")
        for i, split in enumerate(splits[:3]):
            print(f"Parça {i + 1}:")
            print(split.page_content)
            print("-" * 20)
    else:
        print("\nUyarı: 'splits' listesi boş. Döküman yükleme veya bölme işleminde bir sorun olabilir.")
        exit()  # Programdan çık

    vector_store = create_vector_store(splits)
    if not vector_store:
        exit()

    recipe_agent = create_recipe_agent(vector_store)

    question_1 = "İçli köfte nasıl yapılır? Tüm süreci anlat."
    answer_1 = ask_agent(recipe_agent, question_1)
    print(f"Cevap: {answer_1}")
    print("=" * 50)

    question_2 = "Mantı için hangi malzemeler gerekir?"
    answer_2 = ask_agent(recipe_agent, question_2)
    print(f"Cevap: {answer_2}")
    print("=" * 50)

    question_3 = "Elimde kıyma, kimyon ve karabiber var hangi yemekler yapılır?"
    answer_3 = ask_agent(recipe_agent, question_3)
    print(f"Cevap: {answer_3}")
    print("=" * 50)
    
    question_4 = "Sodalı köftenin yanında iyi gidecek bir yemek önerir misin?"
    answer_4 = ask_agent(recipe_agent, question_4)
    print(f"Cevap: {answer_4}")
    print("=" * 50)
