# Türkçe Tarif Defteri ChatBotu

Retrieval-Augmented Generation (RAG) ile geliştirilmiş 5252 adet yemek tarifini 14235 vektöre ayırarak çalışan bir chatbot.
Bugun hangi yemeği yapmalıyım?
"X" yemeği nasıl yapılır?
"X" yemeği için hangi malzemeler gereklidir?
"X" malzemeleri ile hangi yemeği yapabilirim?
"X" yemeğinin yanında ne iyi gider? (Eğer beklentiniz bir ana yemekse ihtiyacınızı karşılamayabilir! Servis için öneri verir.)


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

## Gereksinimler
- streamlit
- langchain
- langchain-community
- langchain-google-genai
- chromadb
- pandas
- python-dotenv
- sentence-transformers
- torch
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
24 - 29 satırı yorum satırına alın
30 - 38 i açın
```python
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
```
Tüm veriseti için 
24 - 29 u açın
30 - 38 i yorum satırına alın

## Huggingface.co Linki
https://huggingface.co/spaces/svenes/recipe-chatbot
