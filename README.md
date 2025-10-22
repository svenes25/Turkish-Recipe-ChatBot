<img width="1024" height="321" alt="image" src="https://github.com/user-attachments/assets/d82c52de-34a1-4f9b-83e8-43af03be2ba7" /># Türkçe Tarif Defteri ChatBotu

Retrieval-Augmented Generation (RAG) ile geliştirilmiş 5252 adet yemek tarifini 14235 vektöre ayırarak çalışan bir chatbot.
- Bugun hangi yemeği yapmalıyım?
- "X" yemeği nasıl yapılır?
- "X" yemeği için hangi malzemeler gereklidir?
- "X" malzemeleri ile hangi yemeği yapabilirim?

### Özellikler
- Verisetindeki verileri vektörel veri setine dönüştürür.
- Gelen soru türüne göre size farklı cevaplar üretir. 
- Öğretilenden farklı sorulara cevap veremez.
- Google Gemini 2.0 Flash modelini kullanır.
- 'intfloat/multilingual-e5-large' Embedding modeli.

### Kullanılan Teknolojiler
- Python
- Langchain
- Streamlit
- LLM: Google Gemini (`gemini-2.0-flash`)
- Embeddings: Google `intfloat/multilingual-e5-large`
- Data: Markdown Q&A (`data2.csv`)
- Chroma

## Gereksinimler
- streamlit
- pandas
- python-dotenv
- langchain
- langchain-core
- langchain-community
- langchain-text-splitters
- langchain-google-genai
- langchain_huggingface
- langchain-chroma
- chromadb
- torch
- sentence-transformers
- .env dosyasına GOOGLE_API_KEY oluşturun.

## Kurulum
1. Repoyu klonlayın
   ```bash
   git clone https://github.com/svenes25/Turkish-Recipe-ChatBot
   cd Turkish-Recipe-ChatBot
   ```
2. Venv oluşturun (Önerilen)
   ```bash
   python -m venv venv
   ```
3. Venv'i çalıştırın (Önerilen)
   ```bash
   venv\Scripts\activate
   ```

4. Bağımlılıkları indirin
   ```bash
   pip install -r requirements.txt
   ```

5. API Key'i bağlayın
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

6. Streamlit i başlatın
   ```bash
   streamlit run app.py
   ```
   Otomatik olarak tarayıcınızda sayfa açılacaktır.
   Açılmazsa tarayıcıda `http://localhost:8500` adresine gidin.

## Proje Mimarisi
```
Turkish-Recipe-ChatBot/
├── venv/
├── datav2.csv        
├── app.py                    
├── rag_chatbot.py       
├── requirements.txt          
├── .env                      
└── README.md                
```

## Hızlı Çalıştırma İçin
30'uncu satırı yorum satırına alın
31'inci satırı açın
```python
   #df = pd.read_csv("datav2.csv")
   df = pd.read_csv("datav2.csv", nrows=30)
```
Tüm veriseti için 
30'uncu satırı açın
31'inci satırı yorum satırına alın

## Huggingface.co Linki
https://huggingface.co/spaces/svenes/turkish-recipe-chatbot
 
- DİKKAT RUN TIME ERROR NEDENİYLE 30 TARİF KULLANILMIŞTIR.
- BAZEN CEVAP NEDENİNİ ANLAMADIĞIM ŞEKİLDE BOŞ DÖNEBİLİYOR.
<img width="975" height="758" alt="image" src="https://github.com/user-attachments/assets/37eab62f-688e-42e4-b09c-abdd9da55299" />
