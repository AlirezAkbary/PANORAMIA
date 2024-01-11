import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.audit_model.audit import AuditModel

class GPT2Distinguisher(torch.nn.Module):
    """Mixed Distinguisher Network for text. Takes in raw texts 
    and logits (featurized by side network) to perform classification.
    Raw text is featurized using a GPT2 model."""

    def __init__(
        self,
        logits_dim: int = None,
        num_output_neurons: int = 1,
        gpt_2_embedding_hidden_dim = 768,
        gpt_name: str = 'gpt2'
        ) -> None:
        """Initialization
        
        Parameters
        ----------

        """
        super().__init__()

        self.gpt2 = AutoModel.from_pretrained(gpt_name)

        # defining the last linear layer separately for controlling the parameters more in optimization
        if logits_dim is not None:
            self.logits_linear_head = nn.Parameter(torch.randn(num_output_neurons, logits_dim))
        else:
            logits_dim = 0

        # initializing with zeros for two-phase optimizationn
        self.gpt_linear_head = nn.Parameter(torch.zeros(num_output_neurons, gpt_2_embedding_hidden_dim))

        # output neuron bias
        self.output_neuron_bias = nn.Parameter(torch.empty(num_output_neurons))

        fan_in = gpt_2_embedding_hidden_dim + logits_dim
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.output_neuron_bias, -bound, bound)


    def forward(self, input_ids=None, attention_mask=None,labels=None, logits=None):
        # passing through the gpt model
        gpt_outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)

        # extracting the rightmost hidden features, which obtains the context of the whole sequence
        pooled_features = gpt_outputs[0][:, -1]

        concat_features, concat_linear_head = pooled_features, self.gpt_linear_head
        if logits is not None:
            # concatenating the pooled features and logits 
            concat_features = torch.cat((pooled_features, logits), dim=1)

            # concatenating the weights
            concat_linear_head = torch.cat((self.gpt_linear_head, self.logits_linear_head), dim=1)

        # applying the linear head
        scores = F.linear(concat_features, concat_linear_head, self.output_neuron_bias)

        # probability of positive class
        probs = torch.sigmoid(scores)

        loss = None

        # return loss if labels are provided
        if labels is not None:
            loss_fct = torch.nn.BCELoss()
            loss = loss_fct(probs.view(-1), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=scores,
            hidden_states=gpt_outputs.hidden_states,
            attentions=gpt_outputs.attentions
        )


class LSTMDistinguisher(torch.nn.Module):
    """Mixed Distinguisher Network for text. Takes in raw texts 
    and logits (featurized by side network) to perform classification.
    Raw text is featurized using a GPT2 model."""

    def __init__(self) -> None:
        """Initialization
        
        Parameters
        ----------

        """
        super().__init__()
        raise NotImplementedError

    def forward(self, input_ids=None, attention_mask=None,labels=None, logits=None):
        pass


class TextAttackModel(torch.nn.Module):
    """Attack model (mia or baseline) general pipeline"""
    
    def __init__(self, side_net: AuditModel, net_type: str = 'mix', distinguisher_type: str = 'GPT2Distinguisher') -> None:
        """Attack Model Initialization:
        Sets up the attack network, turns off gradient on logit_net.

        Parameters
        ----------
        side_net: AuditModel
            if net_type is either mix or only logits, a side network needs to be passed.
            side refers to either the target model f, or the helper (embedding) model.
        net_type: str
            mix: use both raw inputs and logits of f (target) or g
            raw:
            logits:
        distinguisher_type: str
            specifies the architecture of the model that processes the input.
        """
        super(TextAttackModel,self).__init__() 
        self.net_type = net_type
        self.side_net = side_net
        self.distinguisher_type = distinguisher_type

        # side_net needs to be passed if self.net_type is not looking only at raw inputs
        # side_net should specify the dimension of the logits (features) it provides
        if self.net_type != "raw":
            assert self.side_net is not None and hasattr(
                self.side_net, "logits_dim"
            ), "side_net is None or logit_net does not have logits_dim attribute"

            assert hasattr(self.side_net, "get_embedding"), "side_net doesn't have get_embedding method"

            # turn off side_net grad
            self.side_net.freeze_weights()

        # get the network model
        self.model = self.setup_network()

        
    def setup_network(self):
        if self.net_type == 'mix':
            if self.distinguisher_type == 'GPT2Distinguisher':
                return GPT2Distinguisher(
                    logits_dim=self.side_net.logits_dim
                )
            elif self.distinguisher_type == 'LSTMDistinguisher':
                return LSTMDistinguisher()
            else:
                raise NotImplementedError
        elif self.net_type == 'raw':
            if self.distinguisher_type == 'GPT2Distinguisher':
                return GPT2Distinguisher()
            elif self.distinguisher_type == 'LSTMDistinguisher':
                return LSTMDistinguisher()
            else:
                raise NotImplementedError
        elif self.net_type == 'logits':
            raise NotImplementedError
        
        raise NotImplementedError

    def forward(self, input_ids=None, attention_mask=None,labels=None): 
        logits = None
        if self.net_type != "raw":
            # labels should be the same as input_ids
            logits = self.side_net.get_embedding(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        
        return self.model(input_ids, attention_mask, labels, logits)
    
    # def generic_step(self, input_ids=None, attention_mask=None, labels=None):
    #     model_outputs = self(input_ids, attention_mask, labels)
        
    #     train_logits = model_outputs.logits

    #     probs = torch.sigmoid(train_logits)


    # def training_step(self, input_ids=None, attention_mask=None, labels=None):
    #     ...
    
    # def validation_step(self, input_ids=None, attention_mask=None, labels=None):
    #     ...
    
    # def test_step(self, input_ids=None, attention_mask=None, labels=None):
    #     ...
    
    # def generic_epoch_end(self, outputs, split: str) -> None:
    #     ...

    # def training_epoch_end(self, outputs):
    #     return self.generic_epoch_end(outputs, "train")
    
    # def validation_epoch_end(self, outputs):
    #     return self.generic_epoch_end(outputs, "validation")
    
    # def test_epoch_end(self, outputs):
    #     return self.generic_epoch_end(outputs, "test")
    



