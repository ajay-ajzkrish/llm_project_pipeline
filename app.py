import streamlit as st 
from transformers import pipeline



model_sentiment = 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_pipeline = pipeline('sentiment-analysis', model = model_sentiment)

model_generation = "distilgpt2"
generation_pipeline = pipeline('text-generation', model = model_generation)

model_summ = "sshleifer/distilbart-cnn-12-6"
summarization_pipeline = pipeline('summarization', model = model_summ)

model_ner = "dbmdz/bert-large-cased-finetuned-conll03-english"
NER_pipeline = pipeline("ner",model = model_ner)



#web app
def main():
    st.title('LLM Project')

    user_input = st.text_area(max_chars=2000,label='Enter Text Here.', label_visibility="visible")

    selected_option = st.selectbox(label='Selection the requirement.', options = ['Sentiment Analysis', 'Generate Text', 'NER','Summarization'])

    submit = st.button("Submit")

    if submit and len(user_input) > 10:
    #   st.write(submit, user_input, selected_option)
        if selected_option == 'Sentiment Analysis':
            sent = sentiment_pipeline(user_input)
            st.write("The sentiment in sentence is ",sent[0]['label'], " with a score of ", sent[0]['score'])
        elif selected_option == 'Generate Text':
            gen = generation_pipeline(user_input)
            st.write(gen[0]['generated_text'])
    
        if selected_option == 'Summarization':
            if len(user_input) <= 150:
                st.write("Please Input Longer Text")
            else:
                summ = summarization_pipeline(user_input)
                st.write(summ[0]['summary_text'])
        elif selected_option == 'NER':
            ner = NER_pipeline(user_input)
            for i in ner:
                st.write(i['entity'], " : ", i['word'])



if __name__ == "__main__":
    main()