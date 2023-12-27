import logging

import torch

from src.audit_model.audit import AuditModel

class GPT2Distinguisher(torch.nn.Module):
    """Mixed Distinguisher Network for text. Takes in raw texts 
    and logits (featurized by side network) to perform classification.
    Raw text is featurized using a GPT2 model."""

    def __init__(self) -> None:
        """Initialization
        
        Parameters
        ----------

        """
        super().__init__()

    def forward(self, input_ids=None, attention_mask=None,labels=None, logits=None):
        pass


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
        distinguisher_type: str
            specifies the architecture of the model that processes the input.
        """
        super(TextAttackModel,self).__init__() 
        self.net_type = net_type
        self.side_net = side_net
        self.distinguisher_type = distinguisher_type

        # side_net needs to be passed if self.net_type is not looking only at raw inputs
        # side_net should specify the dimension of the logits (features) it provides
        if self.net_type != "only_raw_inputs":
            assert self.side_net is not None and hasattr(
                self.side_net, "logit_dim"
            ), "side_net is None or logit_net does not have logit_dim attribute"

            assert hasattr(self.side_net, "get_embedding"), "side_net doesn't have get_embedding method"

            # turn off side_net grad
            self.side_net.freeze_weights()

        # get the network model
        self.model = self.setup_network()

        

    def setup_network(self):
        if self.net_type == 'mix' or 'only_raw_inputs':
            if self.distinguisher_type == 'GPT2Distinguisher':
                return GPT2Distinguisher()
            elif self.distinguisher_type == 'LSTMDistinguisher':
                return LSTMDistinguisher()
            else:
                return NotImplementedError
        elif self.net_type == 'only_logits':
            pass
        
        return NotImplementedError

    def forward(self, input_ids=None, attention_mask=None,labels=None):
        with torch.no_grad():
            self.side_net.get_embedding(input_ids=input_ids, attention_mask=attention_mask, labels=labels)