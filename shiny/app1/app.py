# shiny run shiny\app.py
# rsconnect deploy shiny <PATH> --name ACCOUNT_NAME --title APP_NAME
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import shinyswatch
from scipy.spatial.distance import cosine
# from sentence_transformers import SentenceTransformer
from shiny import *

openai.api_key = "enter_api_key_here"


def jarvis(prompt: str, topic: Union[None, str]) -> str:
    """
    Uses the OpenAI API to generate an AI response to a prompt.

    Args:
        prompt: A string representing the prompt to send to the OpenAI API.
        topic: An optional string to specify the topic for the AI to focus on. Default is None.

    Returns:
        A string representing the AI's generated response.

    """

    # If no topic is specified, use "Random" as the default topic.
    if topic is None:
        topic = "Random"

    # If a topic is specified, include it in the prompt sent to the API.
    if topic == "Random":
        prompt = f"{prompt}"
    else:
        prompt = f"{prompt}. Make sure use the topic of {topic}."

    # Use the OpenAI API to generate a response based on the input prompt.
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Extract the text from the first (and only) choice in the response output.
    ans = response.choices[0]["text"]

    # Return the generated AI response.
    return ans


def chinese_to_english(prompt: str) -> str:
    """
    Uses the OpenAI API to translate Chinese text into English or correct English grammar.

    Args:
        prompt: A string representing either Chinese language text to be translated or English text with incorrect grammar to be corrected.

    Returns:
        A string representing the translated or corrected English text.

    """

    # Construct a prompt that asks the AI to translate the input Chinese text into English, or correct English grammar if the input text is already in English.
    prompt = f"Translate the following Chinese into English: {prompt}. If it's already in English, just correct its grammar."

    # Use the OpenAI API to generate a response based on the input prompt.
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Extract the text from the first (and only) choice in the response output.
    ans = response.choices[0]["text"]

    # Return the generated AI response.
    return ans


def english_to_chinese(prompt: str) -> str:
    """
    Uses the OpenAI API to correct English grammar and translate the corrected English text into Chinese, providing pinyin pronunciation for each Chinese character.

    Args:
        prompt: A string representing English text with incorrect grammar to be corrected and translated into Chinese.

    Returns:
        A string representing the translated Chinese text, along with pinyin pronunciation for each Chinese character.

    """

    # Construct a prompt that asks the AI to correct the input English text's grammar and translate it into Chinese,
    # while also providing pinyin pronunciation for each Chinese character in the output.
    prompt = f"Correct the English grammar. Then translate the English into Chinese: {prompt}. For each of the Chinese characters, please also provide the Chinese pinyin with tones."

    # Use the OpenAI API to generate a response based on the input prompt.
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Extract the text from the first (and only) choice in the response output.
    ans = response.choices[0]["text"]

    # Return the generated AI response.
    return ans


def url_to_image(url: str) -> np.ndarray:
    """
    Downloads an image from a given URL using urllib, and converts it to a numpy array for use with OpenCV.

    Args:
        url: A string representing URL of the image to be downloaded.

    Returns:
        A numpy array representing the downloaded image.

    """

    # Download the image using urllib.
    with urllib.request.urlopen(url) as url_response:
        img_array = bytearray(url_response.read())

    # Convert the byte array to a numpy array for use with OpenCV.
    img = cv2.imdecode(np.asarray(img_array), cv2.IMREAD_UNCHANGED)

    # Return the image as a numpy array.
    return img


def text_to_img(prompt: str) -> np.ndarray:
    response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")

    image_url = response["data"][0]["url"]
    image = url_to_image(image_url)

    return image


def convert_to_list_of_dict(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Reads in a pandas DataFrame and produces a list of dictionaries with two keys each, 'question' and 'answer.'

    Args:
        df: A pandas DataFrame with columns named 'questions' and 'answers'.

    Returns:
        A list of dictionaries, with each dictionary containing a 'question' and 'answer' key-value pair.
    """

    # Initialize an empty list to store the dictionaries
    result = []

    # Loop through each row of the DataFrame
    for index, row in df.iterrows():
        # Create a dictionary with the current question and answer
        qa_dict_quest = {"role": "user", "content": row["questions"]}
        qa_dict_ans = {"role": "assistant", "content": row["answers"]}

        # Add the dictionary to the result list
        result.append(qa_dict_quest)
        result.append(qa_dict_ans)

    # Return the list of dictionaries
    return result


def get_completion_from_messages(
    messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", temperature: float = 0
) -> str:
    """Generates a response based on the given conversation messages using OpenAI's ChatCompletion API.

    Args:
        messages (List[Dict[str, str]]): A list of messages that make up the conversation history.
        model (str, optional): The name of the AI model to use. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): The degree of randomness of the model's output. Defaults to 0.

    Returns:
        str: The response generated by the AI model.
    """

    # Call the OpenAI ChatCompletion API with the provided parameters
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    # Extract and return the text content of the first message in the response choices
    return response.choices[0].message["content"]


def split_paragraph(paragraph: str) -> str:
    """
    Takes a long paragraph as input, splits it every 50 characters and adds a newline character "\n"
    so that each line is no more than 50 characters long.

    :param paragraph: The long paragraph to be split.
    :type paragraph: str
    :return: A string with newline characters added after every 50 characters.
    :rtype: str
    """
    sentences = paragraph.split(". ")
    new_paragraph = ""
    for sentence in sentences:
        words = sentence.split()
        new_sentence = ""
        for word in words:
            if len(new_sentence + word) > 40:
                new_paragraph += new_sentence.strip() + "\n"
                new_sentence = ""
            new_sentence += word + " "
        new_paragraph += new_sentence.strip() + ". "
    return new_paragraph[:-2]  # Remove the last period and space


def calculate_cosine_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the cosine similarity between two sentences.
    
    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.
    
    Returns:
        float: The cosine similarity between the two sentences, represented as a float value between 0 and 1.
    """
    # Tokenize the sentences into words
    words1 = sentence1.lower().split()
    words2 = sentence2.lower().split()
    
    # Create a set of unique words from both sentences
    unique_words = set(words1 + words2)
    
    # Create a frequency vector for each sentence
    freq_vector1 = np.array([words1.count(word) for word in unique_words])
    freq_vector2 = np.array([words2.count(word) for word in unique_words])
    
    # Calculate the cosine similarity between the frequency vectors
    similarity = 1 - cosine(freq_vector1, freq_vector2)
    
    return similarity


def calculate_sts_score(sentence1: str, sentence2: str) -> float:
    model = SentenceTransformer(
        "paraphrase-MiniLM-L6-v2"
    )  # Load a pre-trained STS model

    # Compute sentence embeddings
    embedding1 = model.encode([sentence1])[0]  # Flatten the embedding array
    embedding2 = model.encode([sentence2])[0]  # Flatten the embedding array

    # Calculate cosine similarity between the embeddings
    similarity_score = 1 - cosine(embedding1, embedding2)

    return similarity_score


def add_dist_score_column(dataframe: pd.DataFrame, sentence: str) -> pd.DataFrame:
    dataframe["sts_score"] = dataframe["questions"].apply(
        lambda x: calculate_cosine_similarity(x, sentence)
    )
    sorted_dataframe = dataframe.sort_values(by="sts_score", ascending=False)

    return sorted_dataframe.iloc[:15,:]


app_ui = ui.page_fluid(
    # style ----
    # Available themes:
    #  cerulean, cosmo, cyborg, darkly, flatly, journal, litera, lumen, lux,
    #  materia, minty, morph, pulse, quartz, sandstone, simplex, sketchy, slate,
    #  solar, spacelab, superhero, united, vapor, yeti, zephyr
    shinyswatch.theme.cerulean(),
    ui.navset_tab(
        # elements ----
        ui.nav(
            "Writing",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_select(
                        "topic",
                        "Select input",
                        {
                            "Random": "Random",
                            "History and literature vs. science and mathematics": "History and literature vs. science and mathematics",
                            "Mandatory vs. optional class attendance": "Mandatory vs. optional class attendance",
                            "Visiting museums while traveling": "Visiting museums while traveling",
                            "Online education vs. traditional education": "Online education vs. traditional education",
                            "Co-ed vs. single-gender schools": "Co-ed vs. single-gender schools",
                            "Learning a foreign language": "Learning a foreign language",
                            "Effects of technology on communication": "Effects of technology on communication",
                            "Advantages and disadvantages of living in a city or countryside": "Advantages and disadvantages of living in a city or countryside",
                            "Importance of art education": "Importance of art education",
                            "Benefits of group study vs. individual study": "Benefits of group study vs. individual study",
                        },
                    ),
                    ui.row(
                        ui.column(
                            8,
                            ui.div(
                                ui.input_action_button("btn", "Create a problem!"),
                            ),
                        )
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                ui.input_action_button(
                                    "btnoverall", "Overall Performance"
                                ),
                            ),
                        )
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                ui.input_action_button("btnsubmit", "Check grammar!"),
                            ),
                        )
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                ui.input_action_button("btnfeedback", "Feedback!"),
                            ),
                        )
                    ),
                ),
                ui.panel_main(
                    ui.h2(
                        {"style": "text-align: center;"},
                        "English as Second Language: Writing",
                    ),
                    ui.h3({"style": "text-align: left;"}, "Essay Topic:"),
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                {"class": "app-col"}, ui.output_text_verbatim("ques")
                            ),
                        )
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                {"class": "app-col"},
                                ui.input_text_area(
                                    "x",
                                    "Text input",
                                    width="100%",
                                    placeholder="I had a an awesome day today! I love it!",
                                ),
                            ),
                        )
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                {"class": "app-col"}, ui.output_text_verbatim("overall")
                            ),
                        )
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                {"class": "app-col"}, ui.output_text_verbatim("txt")
                            ),
                        )
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                {"class": "app-col"},
                                ui.output_text_verbatim("feedback"),
                            ),
                        )
                    ),
                ),
            ),
        ),
        ui.nav(
            "Image Generator",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.row(
                        ui.column(
                            8,
                            ui.div(
                                ui.input_action_button(
                                    "chinese_to_english",
                                    "Type your idea, and create an image!",
                                ),
                            ),
                        )
                    ),
                ),
                ui.panel_main(
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                {"class": "app-col"},
                                ui.input_text_area(
                                    "chinese_to_english_input",
                                    "Enter Chinese or English / 输入中英文即可",
                                    width="100%",
                                    placeholder="请输入中文描述一张你脑海里的图片",
                                ),
                            ),
                        )
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                {"class": "app-col"},
                                ui.output_text_verbatim("chinese_to_english_output"),
                            ),
                        )
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                {"class": "app-col"},
                                ui.output_plot("chinese_to_english_output_plot"),
                            ),
                        )
                    ),
                ),
            ),
        ),
        ui.nav(
            "Learn Chinese",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.row(
                        ui.column(
                            8,
                            ui.div(
                                ui.input_action_button(
                                    "english_to_chinese_btn",
                                    "Type what you want to say in English!",
                                )
                            ),
                        )
                    )
                ),
                ui.panel_main(
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                {"class": "app-col"},
                                ui.input_text_area(
                                    "english_to_chinese_input",
                                    "Enter English here:",
                                    width="100%",
                                    placeholder="Hello, my name is Frannie!",
                                ),
                            ),
                        )
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                {"class": "app-col"},
                                ui.output_text_verbatim(
                                    "english_to_chinese_input_output"
                                ),
                            ),
                        )
                    ),
                ),
            ),
        ),
        ui.nav(
            "Clinical Trial ChatBot",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.row(
                        ui.column(
                            8,
                            ui.div(
                                ui.input_action_button(
                                    "clinical_trials_chatbot_btn",
                                    "Ask me anything!",
                                )
                            ),
                        )
                    )
                ),
                ui.panel_main(
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                {"class": "app-col"},
                                ui.input_text_area(
                                    "clinical_trials_chatbot_input",
                                    "Enter your question here regarding clinical trials:",
                                    width="100%",
                                    placeholder="What is the purpose of a clinical trial?",
                                ),
                            ),
                        )
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.div(
                                {"class": "app-col"},
                                ui.output_text_verbatim(
                                    "clinical_trials_chatbot_input_output"
                                ),
                            ),
                        )
                    ),
                ),
            ),
        ),
    ),
)


def server(input, output, session):
    # The @reactive.event() causes the function to run only when input.btn is
    # invalidated.
    @reactive.Effect
    @reactive.event(input.btn)
    def _():
        print(f"You clicked the button!")
        # You can do other things here, like write data to disk.

    # This output updates only when input.btn is invalidated.
    @output
    @render.text
    @reactive.event(input.btn)
    def ques():
        output = jarvis(
            "Create a random English essay topic. Make sure to list out requirements of 500 words limit.",
            topic=input.topic(),
        )
        return f"Topic: {output}"

    @output
    @render.text
    @reactive.event(input.btnoverall)
    def overall():
        output = jarvis(
            f"Provide an overall analysis of the following essay: {input.x()}. First, give a letter grade from A to D. A is the best and D is the worst. Then provide a tabular view of different perspective of the writing including the following: 1) did the essay address the question above (this is high priority, so make sure this affects the letter grade the most), 2) is the thesis clear?, 3) is the logic and organization clear?, 4) is there topic sentence following with supporting sentence?, 5) how is the transition in between paragraphs?, 6) how is the conclusion statement? Provide the answer in both English and Chinese.",
            topic=None,
        )
        return f"Overall: {output}"

    @output
    @render.text
    @reactive.event(input.btnsubmit)
    def txt():
        output = jarvis(
            f"Check the grammar mistakes of the following: {input.x()}", topic=None
        )
        return f"Grammar: {output}"

    @output
    @render.text
    @reactive.event(input.btnfeedback)
    def feedback():
        output = jarvis(
            f"Did the above prompt answer the question before? Provide detailed critics and explain why. Please write the answer in both English and Chinese.",
            topic=None,
        )
        return f"Feedback: {output}"

    @output
    @render.text
    @reactive.event(input.chinese_to_english)
    def chinese_to_english_output():
        output = chinese_to_english(
            prompt=input.chinese_to_english_input(),
        )
        return f"Processed prompt in English: {output}"

    @output
    @render.plot
    @reactive.event(input.chinese_to_english)
    def chinese_to_english_output_plot():
        output = chinese_to_english(
            prompt=input.chinese_to_english_input(),
        )
        img = text_to_img(output)

        return plt.imshow(img[:, :, ::-1])

    @output
    @render.text
    @reactive.event(input.english_to_chinese_btn)
    def english_to_chinese_input_output():
        output = english_to_chinese(
            prompt=input.english_to_chinese_input(),
        )
        return f"Processed prompt in English: {output}"

    x = reactive.Value("Conversation starts here: \n \n")

    @reactive.Effect
    @reactive.event(input.clinical_trials_chatbot_btn)
    def _():
        infile = Path(__file__).parent / "clinical_trials_qa.csv"
        df = pd.read_csv(infile)
        # Use the DataFrame's to_html() function to convert it to an HTML table, and
        # then wrap with ui.HTML() so Shiny knows to treat it as raw HTML.
        user_question = input.clinical_trials_chatbot_input()
        df_screened_by_dist_score = add_dist_score_column(df, user_question)
        qa_pairs = convert_to_list_of_dict(df_screened_by_dist_score)
        qa_pairs.append({"role": "user", "content": user_question})
        response = get_completion_from_messages(qa_pairs, temperature=1)
        x.set(
            x()
            + " \n\nUser: "
            + user_question
            + " \nAI Doctor: "
            + split_paragraph(response)
            + "\n\n"
        )

    @output
    @render.text
    @reactive.event(input.clinical_trials_chatbot_btn)
    def clinical_trials_chatbot_input_output():
        return str(x())


app = App(app_ui, server, debug=True)
