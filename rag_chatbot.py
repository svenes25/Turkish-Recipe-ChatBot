import os
import sys
import traceback
from dotenv import load_dotenv
import pandas as pd
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_agent

# ==================== TEMEL BİLGİLER ====================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY)


# ==================== 1. VERİ YÜKLEME & PARÇALAMA ====================

def load_and_split_documents():
    try:
        # df = pd.read_csv("datav2.csv")
        df = pd.read_csv("datav2.csv", nrows=10)
        print("10 satırı başarıyla yüklendi.")
        print("=== YÜKLENEN SATIRLAR (ÖN İZLEME) ===")
        print(df.to_string(index=False))  # tüm sütunları düzgün şekilde göster
        print("=====================================\n")
    except FileNotFoundError:
        print("Hata: 'datav2.csv' dosyası bulunamadı.")
        return []
    except Exception as e:
        print(f"CSV yükleme hatası: {e}")
        traceback.print_exc()
        return []

    docs = []
    for _, row in df.iterrows():
        title = row.get("Title", "Bilinmiyor")
        content = (
            f"Yemek Adı: {title}\n\n"
            f"Malzemeler: {row.get('Materials', 'Yok')}\n\n"
            f"Tarif: {row.get('How-to-do', 'Yok')}"
        )
        docs.append(Document(page_content=content, metadata={"title": title}))

    print(f"{len(docs)} adet tarif dökümanı oluşturuldu.")

    # Metni parçalara ayır
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    try:
        splits = text_splitter.split_documents(docs)
    except Exception as e:
        print("Text splitter sırasında hata:", e)
        traceback.print_exc()
        return []

    print(f"Toplam {len(splits)} adet metin parçasına bölündü.")
    return splits


# ==================== 2. VEKTÖR DEPOSU OLUŞTURMA ====================

def create_vector_store(splits, persist: bool = False):
    if not splits:
        print("Uyarı: 'splits' listesi boş. Vektör veritabanı oluşturulamadı.")
        return None

    try:
        print(f"{len(splits)} dökümanla vektör veritabanı oluşturuluyor...")
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large", model_kwargs={"device": "cpu"})
        try:
            vector_store = Chroma.from_documents(documents=splits, embedding=embeddings,
                                                 persist_directory="./chroma_store" if persist else None)
        except TypeError:
            vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        print("Chroma Vektör Deposu başarıyla oluşturuldu!")
        return vector_store
    except Exception as e:
        print(f"Vektör veritabanı oluşturulamadı: {e}")
        traceback.print_exc()
        return None


# ==================== 3. RAG ZİNCİRİ (retrieve tool + agent) ====================

def make_retrieve_context_tool(vector_store):
    from langchain_core.tools import tool
    import traceback

    @tool(description="Kullanıcının sorusuna uygun tarifleri ve malzemeleri getirir.")
    def retrieve_context(question: str) -> str:
        if vector_store is None:
            return "Vektör veritabanı bulunamadı!"
        try:
            docs = vector_store.similarity_search(question, k=5)
        except Exception as e:
            print("Similarity search hatası:", e)
            traceback.print_exc()
            return "Veritabanı araması sırasında hata oluştu."

        if not docs:
            return "Veri bulunamadı."

        composed = ""
        for doc in docs:
            title = doc.metadata.get("title", "Bilinmiyor")
            composed += (
                f"\n🍽️ Yemek Adı: {title}\n"
                f"🧂 Malzemeler: {doc.page_content.split('Malzemeler: ')[-1].split('Tarif:')[0].strip()}\n"
                f"👨‍🍳 Tarif: {doc.page_content.split('Tarif: ')[-1].strip()}\n"
                "──────────────────────────────\n"
            )
        return composed

    return retrieve_context


def create_rag_chain(vector_store):
    from rag_chatbot import create_agent, LLM
    import traceback

    system_prompt = (
        "Sen PROFESYONEL BİR ŞEFSİN ve bir yemek tarifleri uzmanısın. "
        "Kullanıcı sorularını yanıtlamak için aşağıdaki 'bağlam' kısmını kullanmalısın. "
        "Kapsamlı, adım adım ve samimi cevaplar ver. "
        "Eğer bağlamda soruyla ilgili bilgi yoksa, "
        "\"Üzgünüm, veri setimde bu tarife dair bir bilgi bulunmamaktadır.\" diye cevap ver.\n\n"
        "Ayrıca sadece yemek tarifi değil, aşağıdaki türden sorulara da yanıt verebilmelisin:\n"
        "DİL KURALLARI:\n"
        "- Türkçe ek ve kök analizine dikkat et.\n"
        "- Fiillerin çekimli halleri (örneğin 'yapmak', 'yapıyorum', 'yapayım', 'yapabilir miyim', 'yapılır mı') "
        "aynı kökten ('yapmak') türediği için hepsi aynı anlam kategorisine ait olarak değerlendirilmelidir.\n"
        "- Bu kural sadece fiiller için değil, isim-fiil ve sıfat-fiil türevleri için de geçerlidir.\n"
        "- Kullanıcı sorusundaki fiil veya isimleri kök hâline indir ve anlamı bu köke göre eşleştir.\n"
        "- Bugün hangi yemeği yapmalıyım?\n"
        "- 'X' yemeği nasıl yapılır?\n"
        "- 'X' yemeği için hangi malzemeler gereklidir?\n"
        "- 'X' malzemeleri ile hangi yemeği yapabilirim?\n"
        "- 'X' yemeğinin yanında ne iyi gider?\n\n"
        "BAĞLAM:\n{context}"
    )
    retrieve_tool = make_retrieve_context_tool(vector_store)

    try:
        rag_chain = create_agent(tools=[retrieve_tool], model=LLM, system_prompt=system_prompt)
    except Exception as e:
        print("RAG chain oluşturulurken hata:", e)
        traceback.print_exc()
        return None

    return rag_chain


def run_rag_simple(rag_chain, question: str, verbose: bool = False) -> str:
    """RAG zincirinden gelen cevabı yakalayıp düzgün biçimde döndürür."""
    full_response = ""
    inputs = {"messages": [{"role": "user", "content": question}]}

    try:
        for event in rag_chain.stream(inputs, stream_mode="updates"):
            print(event)
            text_part = ""
            if isinstance(event, str):
                text_part = event
            elif isinstance(event, dict):
                if "text" in event:
                    text_part = event["text"]
                elif "output_text" in event:
                    text_part = event["output_text"]
                elif "output" in event and isinstance(event["output"], dict):
                    content = event["output"].get("content")
                    if isinstance(content, str):
                        text_part = content
                    elif isinstance(content, list):
                        text_part = "".join(
                            c.get("text", "") for c in content if isinstance(c, dict)
                        )
                elif "model" in event and "messages" in event["model"]:
                    msg = event["model"]["messages"][0]
                    if hasattr(msg, "content") and isinstance(msg.content, list):
                        text_part = "".join(
                            c["text"] for c in msg.content if isinstance(c, dict) and "text" in c
                        )
            if text_part:
                full_response += text_part
                if verbose:
                    print(text_part, end="", flush=True)

        print("\n" + "-" * 60)
        print("Cevap tamamlandı.")
        print("-" * 60)
        return full_response.strip()

    except Exception as e:
        print(f"run_rag_simple hatası: {e}")
        traceback.print_exc()
        return ""
