import streamlit as st
import pandas as pd
import os
import tempfile
import requests
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import json
import base64
import re

def generate_chart_html(langchain_response):
    chart_data = [['Role', 'Edits']]
    for key, value in langchain_response.items():
        role = key[1]
        edits = value
        chart_data.append([role, edits])

    chart_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
        <script type="text/javascript">
        google.charts.load("current", {{packages: ["corechart"]}});
        google.charts.setOnLoadCallback(drawChart);

        function drawChart() {{
            var data = google.visualization.arrayToDataTable({json.dumps(chart_data)});

            var options = {{
                title: 'Edits Distribution Chart',
                hAxis: {{title: 'Role'}},
                vAxis: {{title: 'Edits'}}
            }};

            var chart = new google.visualization.ColumnChart(document.getElementById('chart_div'));

            chart.draw(data, options);
        }}
        </script>
    </head>
    <body>
        <div id="chart_div" style="width: 800px; height: 400px;"></div>
    </body>
    </html>
    """
    return chart_code

def extract_json_data(response):
    match = re.search(r'{.*}', response)
    if match:
        json_data = match.group(0)
        return json_data
    return None

def convert_response_to_json(response):
    converted_data = {
        'top_journals': [
            {'journal': 'ANZF', 'edits': 1994},
            {'journal': 'CAPR', 'edits': 1944},
            {'journal': 'CAJE', 'edits': 1076},
            {'journal': 'MAPS', 'edits': 977},
            {'journal': 'CJCE', 'edits': 975}
        ]
    }
    return json.dumps(converted_data)

def run_agents(openai_agent, csv_data, user_question):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv_file:
        csv_data.to_csv(temp_csv_file.name, index=False)

    agent = create_csv_agent(openai_agent, temp_csv_file.name, verbose=True)
    response = agent.run(user_question)

    os.remove(temp_csv_file.name)

    return response

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

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

                if response_from_langchain:


                    import openai

                    # Set your OpenAI API key
                    openai.api_key = "OPENAI_API_KEY"  # Replace with your API key

                    # ...

                    if response_from_langchain:
                        # Define a conversation with a system message (optional) and a user message
                        conversation = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": response_from_langchain},
                            {"role": "user", "content": user_question}
                        ]

                        # Generate a response from ChatGPT
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=conversation,
                            max_tokens=150  # Adjust as needed
                        )

                        chatgpt_response_text = response.choices[0].message.content
                        st.write("Response from ChatGPT:")
                        st.write(chatgpt_response_text)


                else:
                    st.error("No response from Langchain")

if __name__ == "__main__":
    main()
