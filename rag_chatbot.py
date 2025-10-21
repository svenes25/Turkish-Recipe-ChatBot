import pandas as pd
from dotenv import load_dotenv
import os
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.tools import tool
from langchain.agents import create_agent
from functools import partial
from langchain_huggingface import HuggingFaceEmbeddings 
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

    # DİKKAT: LangChainDeprecationWarning'i gidermek için HuggingFaceEmbeddings buraya taşındı.
    embeddings = HuggingFaceEmbeddings( 
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': device} 
    )

    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)

    return vector_store

# --- AGENT YAKLAŞIMI İÇİN TOOL TANIMI ---
# Bu, Tool olarak kullanılacak ham fonksiyondur.
@tool
def retrieve_context(query: str, vector_store: Chroma):
    """Yemek tarifleriyle ilgili bir sorguyu yanıtlamaya yardımcı olacak bilgileri getirir."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    
    serialized = "\n\n".join(
        (f"YEMEK ADI: {doc.metadata.get('title', 'N/A')}\nDETAYLAR: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized


def create_recipe_agent(vector_store):
    """
    Sadece Runnable Agent nesnesini döndürür.
    """
    
    # Hata Düzeltmesi: partial ile bağlanan fonksiyonu, bir lambda ile tekrar sarmalayarak
    # 'callable' hatasının önüne geçiyoruz. Lambda, LangChain Tool decorator'ı için daha güvenilirdir.
    retrieval_func = partial(retrieve_context, vector_store=vector_store)
    
    # Lambda ile sarmalama ve Tool olarak sunma
    # Lambda, dinamik olarak oluşturulmuş bir fonksiyon olduğu için 'callable' hatası vermeyecektir.
    retrieval_tool = tool(
        lambda query: retrieval_func(query), # Lambda, query'yi alır ve partial fonksiyonu çağırır
        name="retrieve_recipe_info", 
        description="Kullanıcının sorgusuyla ilgili yemek tarifleri ve malzemeleri getirir. Yemek adlarını, malzemeleri veya nasıl yapılacağını öğrenmek için kullanın."
    )

    tools = [retrieval_tool]
    
    system_prompt = (
        "SEN PROFESYONEL BİR ŞEFSİN. Kullanıcı sorularını yanıtlamak için daima 'retrieve_recipe_info' "
        "adlı aracı kullanmalısın. Gerekirse aracı birden fazla kez çağır. Sadece aracın döndürdüğü "
        "bilgilere dayanarak, detaylı ve kibar bir şekilde cevap ver. Eğer araç bilgi döndürmezse, "
        "\"Üzgünüm, veri setimde bu tarife dair bir bilgi bulunmamaktadır.\" diye cevap ver."
    )
    
    # Agent'ı bir Runnable olarak döndürür
    agent = create_agent(LLM, tools, prompt=system_prompt)
    
    return agent

def stream_agent_answer(agent, question):
    """Oluşturulan Agent'ın akış (stream) çıktısını işler ve yazdırır."""
    print(f"\nSoru: {question}")
    stream_iterator = agent.stream(
        {"input": question}, 
        stream_mode="values"
    )

    print("-" * 50)
    print("AGENT ÇALIŞMA ADIMLARI (STREAMING):")
    print("-" * 50)

    for event in stream_iterator:
        if "messages" in event:
            message = event["messages"][-1]
            if message.tool_calls:
                print(f"-> ARAÇ ÇAĞRISI: {message.tool_calls[0]['name']} (Sorgu: {message.tool_calls[0]['args']['query']})")
            elif message.content:
                if message.tool_call_id:
                    print(f"-> GÖZLEM (Tool Output): {message.content[:100]}...")
                else:
                    # Bu nihai cevaptır
                    print(f"\n--- SON CEVAP ---")
                    print(message.content)
                    print("-----------------\n")

if __name__ == "__main__":
    splits = load_and_split_documents()

    if splits:
        print("\n--- İlk 10 Döküman Parçası ---\n")
        for i, split in enumerate(splits[:10]):
            print(f"Parça {i + 1}:")
            print(split.page_content)
            print("-" * 20)
    else:
        print("\nUYARI: 'splits' listesi boş. Lütfen 'datav2.csv' dosyasını kontrol edin.")
        exit()  # Programdan çık

    vector_store = create_vector_store(splits)
    if not vector_store:
        exit()
    recipe_agent = create_recipe_agent(vector_store)
    stream_agent_answer(recipe_agent, "İçli köfte nasıl yapılır? Tüm süreci anlat.")
    stream_agent_answer(recipe_agent, "Mantı için hangi malzemeler gerekir?")
    stream_agent_answer(recipe_agent, "Elimde kıyma, kimyon ve karabiber var hangi yemekler yapılır?")
    stream_agent_answer(recipe_agent, "Sodalı köftenin yanında iyi gidecek bir yemek önerir misin?")
    print("Program sonlandı.")
