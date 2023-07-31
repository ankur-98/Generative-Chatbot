import torch
import transformers

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List


TOPIC_NUM_LABELS = 10
ACT_NUM_LABELS = 5
EMOTION_NUM_LABELS = 7


class StateTracking(nn.Module):
    def __init__(self, embedding_dims, n_layers, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.embedding_dims = embedding_dims
        self.sequence_tracker = nn.LSTM(self.embedding_dims, self.embedding_dims, n_layers, batch_first=True, bidirectional=True)
        self.state_classifier = nn.Linear(self.embedding_dims * 2, num_labels, bias=True)
        self.loss_fct = CrossEntropyLoss()

    def forward(self, inputs, labels=None):
        _, _, embedding_dims = inputs.shape
        assert (embedding_dims==self.embedding_dims), f"Input embedding dims not equal to the model input dims. {embedding_dims}=={self.embedding_dims}"
        
        sequence_output, _ = self.sequence_tracker(inputs)
        return self.state_classifier(sequence_output)


class RobertaForCausalLMwithParallelStateTracking(transformers.RobertaForCausalLM):
    """
    Ref: https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/roberta/modeling_roberta.py
    """
    def __init__(self, config):
        super().__init__(config)
        self.topic_num_labels = TOPIC_NUM_LABELS
        self.act_num_labels = ACT_NUM_LABELS
        self.emotion_num_labels = EMOTION_NUM_LABELS
        self.topic_state = StateTracking(config.hidden_size, 4, self.topic_num_labels)
        self.act_state = StateTracking(config.hidden_size, 4, self.act_num_labels)
        self.emotion_state = StateTracking(config.hidden_size, 4, self.emotion_num_labels)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        topic_state_labels: Optional[torch.LongTensor] = None,
        act_state_labels: Optional[torch.LongTensor] = None,
        emotion_state_labels: Optional[torch.LongTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RobertaForCausalLM, AutoConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        >>> config = AutoConfig.from_pretrained("roberta-base")
        >>> config.is_decoder = True
        >>> model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            generation_loss = loss_fct(shifted_logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss = generation_loss.clone()
        
            if topic_state_labels is not None:
                topic_state_logits = self.topic_state(outputs[0], topic_state_labels)
                topic_state_loss = loss_fct(
                    topic_state_logits.permute(0, 2, 1), 
                    topic_state_labels.permute(0, 1)
                )                
                loss += topic_state_loss
            if act_state_labels is not None:
                act_state_logits = self.act_state(outputs[0], act_state_labels)
                act_state_loss = loss_fct(
                    act_state_logits.permute(0, 2, 1), 
                    act_state_labels.permute(0, 1)
                )
                loss += act_state_loss
            if emotion_state_labels is not None:
                emotion_state_logits = self.emotion_state(outputs[0], emotion_state_labels)
                emotion_state_loss = loss_fct(
                    emotion_state_logits.permute(0, 2, 1), 
                    emotion_state_labels.permute(0, 1)
                )
                loss += emotion_state_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class OPTForCausalLMwithParallelStateTracking(transformers.OPTForCausalLM):
    """
    Ref: https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/opt/modeling_opt.py
    """
    def __init__(self, config):
        super().__init__(config)
        self.topic_num_labels = TOPIC_NUM_LABELS
        self.act_num_labels = ACT_NUM_LABELS
        self.emotion_num_labels = EMOTION_NUM_LABELS
        self.topic_state = StateTracking(config.word_embed_proj_dim, 4, self.topic_num_labels)
        self.act_state = StateTracking(config.word_embed_proj_dim, 4, self.act_num_labels)
        self.emotion_state = StateTracking(config.word_embed_proj_dim, 4, self.emotion_num_labels)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        topic_state_labels: Optional[torch.LongTensor] = None,
        act_state_labels: Optional[torch.LongTensor] = None,
        emotion_state_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            generation_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            loss = generation_loss.clone()
        
            if topic_state_labels is not None:
                topic_state_logits = self.topic_state(outputs[0], topic_state_labels)
                topic_state_loss = loss_fct(
                    topic_state_logits.permute(0, 2, 1), 
                    topic_state_labels.permute(0, 1)
                )                
                loss += topic_state_loss
            if act_state_labels is not None:
                act_state_logits = self.act_state(outputs[0], act_state_labels)
                act_state_loss = loss_fct(
                    act_state_logits.permute(0, 2, 1), 
                    act_state_labels.permute(0, 1)
                )
                loss += act_state_loss
            if emotion_state_labels is not None:
                emotion_state_logits = self.emotion_state(outputs[0], emotion_state_labels)
                emotion_state_loss = loss_fct(
                    emotion_state_logits.permute(0, 2, 1), 
                    emotion_state_labels.permute(0, 1)
                )
                loss += emotion_state_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
