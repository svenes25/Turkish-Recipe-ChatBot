import pandas as pd
from dotenv import load_dotenv
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import torch
print(torch.__version__)
print(torch.version.cuda)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.")

def load_and_split_documents():
    """Yerel bir dosyadan (CSV) veri setini yükler ve parçalara ayırır."""
    file_path = "datav2.csv"
    # try:
        # df = pd.read_csv(file_path)
        # print(f"'{file_path}' dosyası başarıyla yüklendi.")
    # except FileNotFoundError:
        # print(f"Hata: '{file_path}' dosyası bulunamadı. Lütfen aynı dizinde olduğundan emin olun.")
        # return []
    try:
        df = pd.read_csv(file_path,nrows=10)
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

def howto_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
    Sen bir şefsin. Sadece verilen bağlamı kullanarak soruları cevapla.
    Eğer verilen bağlamda cevap yoksa, "Verilen bağlamda bu soruya dair bir bilgi bulunmamaktadır." şeklinde kibarca cevap ver.
    Yemek içeriğinden bahset.
    Tüm süreci bir paragraf olacak şekilde anlat.

    Bağlam: {context}
    Soru: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
def material_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
    Sen bir şefsin. Sadece verilen bağlamı kullanarak soruları cevapla.
    Eğer verilen bağlamda cevap yoksa, "Verilen bağlamda bu soruya dair bir bilgi bulunmamaktadır." şeklinde kibarca cevap ver.
    Yemek materyalinden bahset.
    Materyalleri sırala.
    Eğer isterse yemek adını söyleyerek nasıl yapacağını tarif edeceğini söyle.
    Çok uzun yazma.

    Bağlam: {context}
    Soru: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
def title_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
    SEN PROFESYONEL BİR ŞEFSİN. GÖREVİN, ANA YEMEKLE MÜKEMMEL BİR UYUM SAĞLAYACAK TAMAMLAYICI BİR YAN YEMEĞİ, SADECE VERİLEN BAĞLAMDAKİ DİĞER TARİFLER ARASINDAN SEÇMEKTİR.

1.  **Ana Yemek Analizi:** Kullanıcının sorduğu yemeğin (Sodalı Köfte) lezzet profilini ve dokusunu (örneğin: ağır, etli, yağlı) belirle.
2.  **Yan Yemek Seçimi:** Bağlamda bulunan **DİĞER YEMEK ADLARINI** inceleyerek, analiz ettiğin bu özelliklere **kontrast (zıt)** veya mükemmel bir denge oluşturacak bir TAM TARİF seç. **ASLA sadece servis önerileri (domates, biber gibi malzemeler) önerme. SADECE tam bir yemek adı öner.**
3.  **Açıklama:** Önerdiğin yemeğin (Adını Mutlaka Belirterek), ana yemeği nasıl dengelediğini veya lezzetini nasıl zenginleştirdiğini, bir şef gibi **1-2 cümle** ile açıkla.

**Format Kuralı (ÇOK ÖNEMLİ):** Cevabın, sadece seçtiğin yan yemeğin adı ve 1-2 cümlelik açıklamasından oluşmalıdır. Bağlamdaki alakasız cümleleri (örneğin: "yanına domates koyabilirsiniz") veya kendi analizlerini cevabına DAHİL ETME.

Eğer Bağlam'da sorulan yemeğe dair ana bir bilgi yoksa, "Verilen bağlamda bu soruya dair bir bilgi bulunmamaktadır." şeklinde kibarca cevap ver.

Bağlam: {context}
Soru: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


def mat_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
    Sen bir şefsin. Sadece verilen bağlamı kullanarak soruları cevapla.
    Eğer verilen bağlamda cevap yoksa, "Verilen bağlamda bu soruya dair bir bilgi bulunmamaktadır." şeklinde kibarca cevap ver.
    Verilen materyalleri incele. 
    Datasetinde bulunan herhangi bir tarifin adını ver.
    Çok uzun yazma.

    Bağlam: {context}
    Soru: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def ask_rag_chain(chain, question):
    """Oluşturulan RAG zinciri ile soru sorar."""
    response = chain.invoke({"input": question})
    return response["answer"]


# Ana program akışı
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
    how_chain = howto_rag_chain(vector_store)
    question = "İçli köfte nasıl yapılır?"
    print(f"\nSoru: {question}")
    answer = ask_rag_chain(how_chain, question)
    print(f"Cevap: {answer}")

    material_chain = material_rag_chain(vector_store)
    question_2 = "Mantı için hangi malzemeler gerekir?"
    print(f"\nSoru: {question_2}")
    answer_2 = ask_rag_chain(material_chain, question_2)
    print(f"Cevap: {answer_2}")

    mat_chain = mat_rag_chain(vector_store)
    question_3 = "Elimde kıyma, kimyon ve karabiber var hangi yemekler yapılır?"
    print(f"\nSoru: {question_3}")
    answer_3 = ask_rag_chain(mat_chain, question_3)
    print(f"Cevap: {answer_3}")

    title_chain = title_rag_chain(vector_store)
    question_4 = "Sodalı köftenin yanında ne iyi gider?"
    print(f"\nSoru: {question_4}")
    answer_4 = ask_rag_chain(title_chain, question_4)
    print(f"Cevap: {answer_4}")
