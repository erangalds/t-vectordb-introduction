import streamlit as st
from transformers import pipeline

# Initialize the language model
model = pipeline("text-generation", model="gpt-2")

# Define the Streamlit app
def main():
    st.title("Streamlit Chat Application")
    
    # User input
    with st.form(key='chat_bot_form'):
        user_input = st.text_input()
        submit_button = st.form_submit_button(label='Send')

    if user_input and submit_button:
        # Generate a response
        response = model(user_input, max_length=50, num_return_sequences=1)
        answer = response[0]['generated_text']
        
        # Display the response
        st.write("### Question:")
        st.write(user_input)
        st.write("### Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()
