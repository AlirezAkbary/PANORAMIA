import torch

class AuditModel():
    def __init__(
        self, 
        model: torch.nn.Module
        ) -> None:
        self.model = model
        self.model.eval()
    
    def get_embedding(self, input_ids, attention_mask, labels, batched=False, output_last_hidden_state=False, output_loss=False):
        pass
    
    def freeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad=False
        

    
class AuditModelGPT2CLM(AuditModel):
    def __init__(self, model) -> None:
        super().__init__(model)

    def get_embedding(self, input_ids, attention_mask, labels, batched=False, output_last_hidden_state=False, output_loss=False):
        with torch.no_grad():
            if output_loss:
                if not batched:
                    model_outputs_loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                    return model_outputs_loss
                else:
                    outputs_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                    labels = torch.clone(input_ids)
                    shift_logits = outputs_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    return loss.view(outputs_logits.size(0), -1).mean(dim=1)

            # get the last hidden states of the audit model. size: (batch_num. seq_len, hidden_dim)
            last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]

            #get only the vector corresponding to [CLS] token in the sequence, [NOTE] assuming it has enough information of the whole sequence. size: (batch_num, hidden_dim)
            #last_hidden_state = last_hidden_state[:, 0]
            
            # get the logits of the model. size: (batch_num, num_classes)
            outputs_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

            if output_last_hidden_state:
                return last_hidden_state
            else:  
                return outputs_logits
