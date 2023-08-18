# Generative Chatbot

This repository contains a generative chatbot that takes a conversation history as input and generates a valid response. The dataset used for this task is [Daily Dialogues](http://yanran.li/dailydialog). 

## Overview

The chatbot is designed with the assumption that the first speaker is the user and the second one is the agent. All turns until the agent's last turn are taken as conversation history, and the chatbot predicts the last agent's message. The chatbot uses a transformers model for response generation and a Gradio interface for interaction. This project also includes a bonus feature that leverages the emotion labels provided in the dataset to improve response generation by making the chatbot emotion-aware. 

![DailyDialogue Example](http://yanran.li/images/dailydialog_example_smaller.jpg)

## Installation

To set up the environment for this project, ensure [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed. Then, you can create the Conda environment and install the dependencies with:

   ```bash
   conda env create -f environment.yml
   ```

Activate the environment with:

   ```bash
   conda activate generative_chatbot
   ```

## Usage

To start the Gradio interface, run:

   ```bash
   . ./test.sh
   ```
