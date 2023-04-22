import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from shiny import *
# from shiny.types import NavSetArg
from shiny.ui import h4
import openai
import urllib.request


# shiny run shiny\app.py
# rsconnect deploy shiny <PATH> --name ACCOUNT_NAME --title APP_NAME
openai.api_key = "<ENTER API KEY HERE>"

def jarvis(prompt: str, topic: any) -> str:

    if topic is None:
        topic = "Random"

    if topic == "Random":
        prompt = f"{prompt}"
    else:
        prompt = f"{prompt}. Make sure use the topic of {topic}."

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    ans = response.choices[0]["text"]

    return ans


def chinese_to_english(prompt: str) -> str:

    prompt = f"Translate the following Chinese into English: {prompt}. If it's already in English, just correct it's grammar."

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    ans = response.choices[0]["text"]

    return ans


def url_to_image(url: str) -> np.ndarray:

    # Download the image using urllib
    with urllib.request.urlopen(url) as url_response:
        img_array = bytearray(url_response.read())

    # Convert the byte array to a numpy array for use with OpenCV
    img = cv2.imdecode(np.asarray(img_array), cv2.IMREAD_UNCHANGED)

    # Return the image as a numpy array
    return img


def text_to_img(prompt: str) -> np.ndarray:
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )

    image_url = response['data'][0]['url']
    image = url_to_image(image_url)

    return image

app_ui = ui.page_fluid(
    # style ----
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
                            "Benefits of group study vs. individual study": "Benefits of group study vs. individual study"
                        }),
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
                                ui.input_action_button("btnoverall", "Overall Performance"),
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
                    ui.h2({"style": "text-align: center;"}, "English as Second Language: Writing"),
                    ui.h3({"style": "text-align: left;"}, "Essay Topic:"),
                    ui.row(ui.column(12, ui.div({"class": "app-col"}, ui.output_text_verbatim("ques")))),
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
                    ui.row(ui.column(12, ui.div({"class": "app-col"}, ui.output_text_verbatim("overall")))),
                    ui.row(ui.column(12, ui.div({"class": "app-col"}, ui.output_text_verbatim("txt")))),
                    ui.row(ui.column(12, ui.div({"class": "app-col"}, ui.output_text_verbatim("feedback")))),
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
                                ui.input_action_button("chinese_to_english", "Type your idea, and create an image!"),
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
                    ui.row(ui.column(12, ui.div({"class": "app-col"}, ui.output_text_verbatim("chinese_to_english_output")))),
                    ui.row(ui.column(12, ui.div({"class": "app-col"}, ui.output_plot("chinese_to_english_output_plot")))),

                )
            )),
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
            topic=input.topic())
        return f"Topic: {output}"

    @output
    @render.text
    @reactive.event(input.btnoverall)
    def overall():
        output = jarvis(
            f"Provide an overall analysis of the following essay: {input.x()}. First, give a letter grade from A to D. A is the best and D is the worst. Then provide a tabular view of different perspective of the writing including the following: 1) did the essay address the question above (this is high priority, so make sure this affects the letter grade the most), 2) is the thesis clear?, 3) is the logic and organization clear?, 4) is there topic sentence following with supporting sentence?, 5) how is the transition in between paragraphs?, 6) how is the conclusion statement? Provide the answer in both English and Chinese.",
            topic=None)
        return f"Overall: {output}"

    @output
    @render.text
    @reactive.event(input.btnsubmit)
    def txt():
        output = jarvis(
            f"Check the grammar mistakes of the following: {input.x()}",
            topic=None)
        return f"Grammar: {output}"
    
    @output
    @render.text
    @reactive.event(input.btnfeedback)
    def feedback():
        output = jarvis(
            f"Did the above prompt answer the question before? Provide detailed critics and explain why. Please write the answer in both English and Chinese.",
            topic=None)
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



app = App(app_ui, server, debug=True)
