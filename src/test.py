"""
This script launches a Gradio interface for a conversational chatbot.
The chatbot is powered by a transformers model and can be configured with
various parameters.
"""

import argparse
import gradio
import transformers
import math
import logging
from typing import List

# Set up logging
logging.basicConfig(
    filename='chatbot.log', 
    filemode='w', 
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class ChatAgent:
    """
    A chat agent that can generate responses to user input.
    The agent is powered by a transformers model and maintains a history of
    the conversation.
    """
    def __init__(
        self, model_name_or_path: str, model_max_length: int,
        penalty_alpha: float, repetition_penalty: float, top_k: int,
        temperature: float, num_beams: int, num_beam_groups: int,
        diversity_penalty: float) -> None:
        """
        Initializes the chat agent with the specified transformers model and
        parameters.
        """
        # Load the transformers model
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path, is_decoder=True)
        self.model.config.use_cache = False

        # Adjust the model's max position embeddings if necessary
        orig_ctx_len = getattr(self.model.config, "max_position_embeddings", None)
        if orig_ctx_len and model_max_length > orig_ctx_len:
            scaling_factor = math.ceil(model_max_length / orig_ctx_len)
            self.model.config.rope_scaling = {
                "type": "linear", 
                "factor": scaling_factor
            }

        # Load the tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir="./.cache/training",
            model_max_length=model_max_length,
            truncation_side="left",
            padding=False,
            use_fast=False,
        )

        # Create the text generation pipeline
        self.generator = transformers.pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            min_length=20,
            max_length=model_max_length, 
            penalty_alpha=penalty_alpha,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            temperature=temperature,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            early_stopping=False,
            device="cuda",
            use_cache=True
        )

        # Initialize the conversation history
        self.history = []

    def respond(self, user: str) -> str:
        """
        Generates a response to the given user input and updates the 
        conversation history.
        """
        # Prepare the inputs for the model
        inputs = f" {self.tokenizer.eos_token} ".join(self.history)
        inputs += f" {self.tokenizer.eos_token} " if len(self.history) > 0 else ""
        inputs += f"{user} {self.tokenizer.eos_token} response:"
        
        # Generate the response
        bot = self.generator(user)[0]["generated_text"]
        
        # Log the user input and bot response
        logging.info(f"User: {user}")
        logging.info(f"Bot: {bot}")

        # Update the conversation history
        self.history.extend([user, bot])

        return bot

    def reset(self) -> str:
        """
        Resets the conversation history.
        """
        self.history = []
        return "Chat history cleared!"

def create_agent(
    model_name_or_path: str, model_max_length: int,
    penalty_alpha: float, repetition_penalty: float, top_k: int,
    temperature: float, num_beams: int, num_beam_groups: int,
    diversity_penalty: float, user: str) -> str:
    """
    Creates a ChatAgent and generates a response to the given user input.
    """
    agent = ChatAgent(
        model_name_or_path, model_max_length, penalty_alpha, 
        repetition_penalty, top_k, temperature, num_beams, 
        num_beam_groups, diversity_penalty)
    return agent.respond(user)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999, 
                        help="Port to run the Gradio interface on")
    args = parser.parse_args()
    
    # Define the Gradio interface
    iface = gradio.Interface(
        fn=create_agent,
        inputs=[
            gradio.Dropdown(
                choices=[
                    "./ckpt/state_tracking/opt_350/checkpoint-1500", 
                    "./ckpt/state_tracking/opt_125/checkpoint-1800", 
                    "./ckpt/state_tracking/roberta-base/checkpoint-1800", 
                    "facebook/opt-350m", 
                    "facebook/opt-125m", 
                    "roberta-base"
                ], 
                label="Model Name or Path"
            ),
            gradio.Slider(30, 1024, default=512, label="Model Max Length"),
            gradio.Slider(0, 1, step=0.1, default=0.0, 
                                 label="Penalty Alpha"),
            gradio.Slider(1.0, 10, step=0.1, default=1.0, 
                                 label="Repetition Penalty"),
            gradio.Slider(1, 100, default=50, step=1, label="Top K"),
            gradio.Slider(0, 1, step=0.1, default=1.0, 
                                 label="Temperature"),
            gradio.Slider(1, 10, default=1, label="Num Beams"),
            gradio.Slider(1, 10, default=1, label="Num Beam Groups"),
            gradio.Slider(0, 1, step=0.1, default=0.0, 
                                 label="Diversity Penalty"),
            gradio.Textbox(lines=2, 
                                  placeholder="Enter your message here...", 
                                  label="User")
        ],
        outputs="text",
    )

    # Launch the interface
    iface.launch()
