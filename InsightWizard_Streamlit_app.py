import streamlit as st
import pandas as pd
import os
import tempfile
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import json
import io
import altair as alt
import openai

def run_agents(openai_agent, csv_data, user_question):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv_file:
        csv_data.to_csv(temp_csv_file.name, index=False)

    agent = create_csv_agent(openai_agent, temp_csv_file.name, verbose=True)
    response = agent.run(user_question)

    os.remove(temp_csv_file.name)

    return response

def generate_dynamic_chart(data, chart_title="Dynamic Chart"):
    dfs = []

    def process_data(data, prefix=""):
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}_{key}" if prefix else key
                process_data(value, new_prefix)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_prefix = f"{prefix}_{i}" if prefix else str(i)
                process_data(item, new_prefix)
        else:
            if isinstance(data, str) or isinstance(data, int) or isinstance(data, float):
                df = pd.DataFrame({'Value': [data]})
                df['Role'] = prefix
                dfs.append(df)

    process_data(data)

    if not dfs:
        st.warning("No valid data found for chart.")
        return None

    combined_df = pd.concat(dfs)

    chart = alt.Chart(combined_df).mark_bar().encode(
        x=alt.X('Role:N', title='Category'),
        y=alt.Y('Value:Q', title='Value'),
        color=alt.Color('Role:N', title='Role')
    ).properties(
        title=chart_title,
        width=400
    )

    return chart

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    #st.write(os.getenv("OPENAI_API_KEY"))

    if api_key is None or api_key == "":
        st.error("OPENAI_API_KEY is not set")
        return

    st.set_page_config(page_title="Insight Wizard")
    st.header("Insight Wizard 💡")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        openai_agent = OpenAI(api_key=api_key, temperature=0)
        csv_data = pd.read_csv(csv_file)

        user_question_llm = st.text_input("Ask a question about your CSV: ")
        user_question = st.text_input("Ask a question about your data: ")

        if user_question_llm is not None and user_question_llm != "":
            with st.spinner(text="In progress..."):
                response_from_langchain = run_agents(openai_agent, csv_data, user_question_llm)
                st.write(response_from_langchain)

                if response_from_langchain:
                    openai.api_key = api_key

                    conversation = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": response_from_langchain},
                        {"role": "user", "content": user_question}
                    ]

                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=conversation,
                        max_tokens=250
                    )

                    chatgpt_response_text = response.choices[0].message.content
                    st.write("Response from ChatGPT:")
                    st.write(chatgpt_response_text)

                    user_chart_data = st.text_area("Enter chart data in JSON format:")
                    if st.button("Generate Dynamic Chart"):
                        try:
                            data = json.loads(user_chart_data)
                            if data:
                                chart = generate_dynamic_chart(data)
                                st.altair_chart(chart, use_container_width=True)
                        except json.JSONDecodeError:
                            st.error("Invalid JSON data. Please provide data in a valid JSON format.")
                        except openai.error.OpenAIError as e:  # Update this line
                            st.error(f"Error from OpenAI: {str(e)}")

if __name__ == "__main__":
    main()


