## LG Electronics Toronto AI Lab, Conversational AI Take Home Assignment

# Generative Chatbot
Build a chatbot that takes a conversation history as input and generates a valid response. 
The dataset that is used for this task is Daily Dialogues (http://yanran.li/dailydialog). 

## Assumptions about the project
1. You can assume the first speaker is the user and the second one is the agent
2. You can take all the turns until agent's last turn as conversation history and predict the last agent's message using your model.

An example conversation from this dataset is shown below
![DailyDialogue Example](http://yanran.li/images/dailydialog_example_smaller.jpg)

You will be given 24 hours to complete the task and share the output with Geoff.

## Deliverables
1. A report about your project
2. A project with the following:
   - README 
   - requirements.txt
   - Appropriate project structure
3. A model checkpoint trained using this dataset
   - Important Notes:
     - Please avoid using prompting based solutions. 
     - We need to observe your ideation capability and creativity for designing a solution in this timeframe
4. Inference code to test your checkpoint

## Bonus 
This dataset also provides emotion labels, so you can leverage them to improve response generation by being emotion-aware.
If you completing the bonus part of the assignment, please provide these details in your report as well.
