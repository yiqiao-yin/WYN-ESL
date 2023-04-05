# Shiny GPT-3 Interaction App

This Shiny app provides an interactive interface to communicate with OpenAI's GPT-3 model using the `jarvis` function. The app consists of a frontend for user input and a server-side backend for handling the GPT-3 model interaction.

## Structure

The Shiny app is located in the `shiny` folder in the main Git repository. The main script, `app.py`, is responsible for running the app and contains both the frontend and server-side code.

### Frontend

The frontend provides a user interface for entering text and submitting it to the GPT-3 model. Users can input their queries, and the app will display the model's response.

### Server-side

On the server side, the `jarvis` function makes a call to OpenAI's `open.completion` code, which allows the app to interact with the GPT-3 model. The `jarvis` function receives the user's input from the frontend, processes it, and returns the GPT-3 model's response.

## User Manual

To run the app locally or deploy it, follow the instructions below.

### Running the app locally

1. Ensure that you have Shiny installed on your local machine.
2. Open your terminal or command prompt and navigate to the main Git directory.
3. Run the following command to start the app:

```sh
shiny run shiny /app.py
```

The app should now be running locally, and you can access it through your web browser.

### Deploying the app

To deploy the Shiny app, you'll need to use the `rsconnect` package. Make sure it's installed on your local machine.

1. Open your terminal or command prompt and navigate to the main Git directory.
2. Run the following command to deploy the app, replacing `<PATH>`, `ACCOUNT_NAME`, and `APP_NAME` with the appropriate values:

```sh
rsconnect deploy shiny <PATH> --name ACCOUNT_NAME --title APP_NAME
```

The app should now be deployed and accessible through the specified account and app name.

Once the app is running locally or deployed, you can start interacting with the GPT-3 model. Enter your text or query in the input field provided on the app's frontend, then click the "Submit" button. The app will process your input and display the GPT-3 model's response below the input field.





