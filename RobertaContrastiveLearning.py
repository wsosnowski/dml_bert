import torch
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaClassificationHead
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaContrastiveLearning(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()
        self.class_weight = kwargs.get('class_weights', None)
        self.clf_loss = kwargs.get('clf_loss', None)
        self.beta = kwargs.get('beta', None)
        self.only_cls = kwargs.get('only_cls', None)
        self.extended_inference = kwargs.get('extended_inference', None)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        if self.beta is not None:
            epsilon = self.beta
        else:
            epsilon = 1
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.config is not None:
                    loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                flatten_labels = torch.ones(sequence_output.shape[0] * sequence_output.shape[1]).type(
                    torch.LongTensor).to(loss.device)

                for label in labels.view(-1):
                    for idx, _ in enumerate(range(sequence_output.shape[1])):
                        flatten_labels[idx] = label

                sequence_selected = sequence_output.view(-1, sequence_output.shape[2])
                if self.clf_loss is not None:
                    if self.only_cls:
                        cl_loss = self.clf_loss(sequence_output[:, 0, :], labels.view(-1))
                    else:
                        cl_loss = self.clf_loss(sequence_selected, flatten_labels)

                    loss = epsilon * loss + (1 - epsilon) * cl_loss

            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if self.extended_inference is not None and self.clf_loss is not None:
            sim_to_classes = self.clf_loss.get_logits(sequence_output[:, 0, :])
            softmax = torch.nn.Softmax(dim=1)
            logits = epsilon * softmax(logits) + (1 - epsilon) * softmax(sim_to_classes)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
