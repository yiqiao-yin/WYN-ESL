from typing import List
from shiny import *
from shiny.types import NavSetArg
from shiny.ui import h4
import openai

# shiny run shiny\app.py
# rsconnect deploy shiny <PATH> --name ACCOUNT_NAME --title APP_NAME

def jarvis(prompt: str, topic: any):
    openai.api_key = "<ENTER_API_KEY_HERE>"

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
            "Critical Reasoning",
            "tab b content"),
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


app = App(app_ui, server, debug=True)
