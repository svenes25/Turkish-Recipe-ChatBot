import streamlit as st
from rag_chatbot import load_and_split_documents, create_vector_store, create_rag_chain, run_rag_simple

st.title("👨‍🍳 Yemek Tarifleri Asistan")

# RAG zinciri yükleme
if "rag_chain" not in st.session_state:
    with st.spinner("Yemek tarifleri yükleniyor ve vektörleştiriliyor..."):
        try:
            splits = load_and_split_documents()
            if not splits:
                st.error("Veri seti yüklenirken veya işlenirken bir sorun oluştu. Lütfen konsol çıktılarını kontrol edin.")
            else:
                vector_store = create_vector_store(splits)
                if vector_store:
                    st.session_state.rag_chain = create_rag_chain(vector_store)
                    st.success("Hazır!")
                else:
                    st.error("Vektör veritabanı oluşturulamadı. Lütfen konsol çıktılarını kontrol edin.")
        except Exception as e:
            st.error(f"Hata oluştu: {e}")

# Chat geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

# Önceki mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni soru
if prompt := st.chat_input("Bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if "rag_chain" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Cevap oluşturuluyor..."):
                response = run_rag_simple(st.session_state.rag_chain, prompt, verbose=False)

                if not response:
                    response = "_(Cevap boş döndü)_"

                # 👇 Cevabı ekranda tutmak için state'e kaydet
                st.session_state.messages.append({"role": "assistant", "content": response})

                st.markdown(response)
    else:
        st.warning("Chatbot henüz hazır değil. Lütfen yükleme işleminin tamamlanmasını bekleyin.")