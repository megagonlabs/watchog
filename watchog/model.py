import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}


def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class UnsupCLforTable(nn.Module):
    """ BarlowTwins/SimCLR encoder for contrastive learning
    """
    def __init__(self, hp, device='cuda', lm='roberta'):
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        self.temperature = hp.temperature
        hidden_size = 768

        # projector
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # contrastive
        # self.criterion = nn.CrossEntropyLoss().to(device)
        self.criterion = nn.CrossEntropyLoss().cuda()
        # cls token id
        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp[lm]).cls_token_id


    def info_nce_loss(self, features,
            batch_size,
            n_views):
        """Copied from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        return logits, labels

    def _extract_columns(self, x, z, cls_indices=None):
        """Helper function for extracting column vectors from LM outputs.
        """
        x_flat = x.view(-1)
        column_vectors = z.view((x_flat.shape[0], -1))

        if cls_indices is None:
            indices = [idx for idx, token_id in enumerate(x_flat) \
                if token_id == self.cls_token_id]
        else:
            indices = []
            seq_len = x.shape[-1]
            for rid in range(len(cls_indices)):
                indices += [idx + rid * seq_len for idx in cls_indices[rid]]

        return column_vectors[indices]
    
    def _extract_table(self, x, z, cls_indices=None):
        """Helper function for extracting all column vectors in a table from LM outputs.
        """
        x_flat = x.view(-1)
        column_vectors = z.view((x_flat.shape[0], -1))

        if cls_indices is None:
            indices = [idx for idx, token_id in enumerate(x_flat) \
                if token_id == self.cls_token_id]
        else:
            indices = []
            seq_len = x.shape[-1]
            for rid in range(len(cls_indices)):
                indices += [idx + rid * seq_len for idx in cls_indices[rid]]

        return torch.mean(torch.stack(column_vectors[indices]))


    def inference(self, x):
        """Apply the model on a serialized table.

        Args:
            x (LongTensor): a batch of serialized tables

        Returns:
            Tensor: the column vectors for all tables
        """
        # x = x.to(self.device)
        x = x.cuda()
        z = self.bert(x)[0]
        z = self.projector(z) # optional
        return self._extract_columns(x, z)


    def forward(self, x_ori, x_aug, cls_indices, mode="simclr", task="None"):
        """Apply the model for contrastive learning.

        Args:
            x_ori (LongTensor): the first views of a batch of tables
            x_aug (LongTensor): the second views of a batch of tables
            cls_indices (tuple of List): the cls_token alignment
            mode (str, optional): the name of the contrastive learning algorithm
            task (str, optional): the supervision signal, unsupervised if task == "None"

        Returns:
            Tensor: the loss
        """
        if mode in ["simclr", "barlow_twins"]:
            # pre-training
            # encode
            batch_size = len(x_ori)
            x_ori = x_ori.cuda() # original, (batch_size, seq_len)
            x_aug = x_aug.cuda() # augment, (batch_size, seq_len)

            # encode each table (all columns)
            x = torch.cat((x_ori, x_aug)) # (2*batch_size, seq_len)
            z = self.bert(x)[0] # (2*batch_size, seq_len, hidden_size)

            # assert that x_ori and x_aug have the same number of columns
            z_ori = z[:batch_size] # (batch_size, seq_len, hidden_size)
            z_aug = z[batch_size:] # (batch_size, seq_len, hidden_size)
            
            cls_ori, cls_aug = cls_indices

            z_ori_col = self._extract_columns(x_ori, z_ori, cls_ori) # (total_num_columns, hidden_size)
            z_aug_col = self._extract_columns(x_aug, z_aug, cls_aug) # (total_num_columns, hidden_size)
            assert z_ori_col.shape == z_aug_col.shape

            z_col = torch.cat((z_ori_col, z_aug_col))
            z_col = self.projector(z_col) # (2*total_num_columns, projector_size)

            if mode == "simclr":
                # simclr
                logits, labels = self.info_nce_loss(z_col, len(z_col) // 2, 2)
                loss = self.criterion(logits, labels)                            
                return loss
            
            elif mode == "barlow_twins":
                # barlow twins
                z1 = z[:len(z) // 2]
                z2 = z[len(z) // 2:]

                # empirical cross-correlation matrix
                c = (self.bn(z1).T @ self.bn(z2)) / (len(z1))

                # use --scale-loss to multiply the loss by a constant factor
                on_diag = ((torch.diagonal(c) - 1) ** 2).sum() * self.hp.scale_loss
                off_diag = (off_diagonal(c) ** 2).sum() * self.hp.scale_loss
                loss = on_diag + self.hp.lambd * off_diag
                return loss
       


class SupCLforTable(nn.Module):
    """Supervised contrastive learning encoder for tables.
    """
    def __init__(self, hp, device='cuda', lm='roberta'):
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        self.temperature = hp.temperature
        hidden_size = 768
        # projector
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(hidden_size, affine=False)

        # a fully connected layer for fine tuning
        # self.fc = nn.Linear(hidden_size * 2, 2)

        # contrastive
        self.criterion = nn.CrossEntropyLoss().cuda()

        # cls token id
        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp[lm]).cls_token_id

    def load_from_pretrained_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model = UnsupCLforTable(ckpt['hp'], device=self.device, lm=ckpt['hp'].lm)
        model.load_state_dict(ckpt['model'])
        self.bert = model.bert
        self.cls_token_id = model.cls_token_id
        del model

    def info_nce_loss(self, features,
            batch_size,
            n_views,
            temperature=0.07):
        """Copied from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / temperature
        return logits, labels
    
    def supcon_loss(self, features,
            batch_size,
            n_views,
            signals=None):
        
        features = F.normalize(features, dim=1)
        # discard the main diagonal from both: labels and similarities matrix
        similarity_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        logits = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0].detach()
        if len(signals.shape) == 1:
            mask_similar_class = (signals.unsqueeze(1).repeat(1, signals.shape[0]) == signals).cuda()
        else:
            mask_similar_class = (torch.sum(torch.eq(signals.unsqueeze(1).repeat(1,signals.shape[0],1),
                                                     signals.unsqueeze(0).repeat(signals.shape[0],1,1)), dim=-1) == signals.shape[-1]).cuda()
            
        mask_anchor_out = (1 - torch.eye(logits.shape[0])).cuda()
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        
        # log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        exp_logits = torch.exp(logits) * mask_anchor_out
        log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = -torch.mean(supervised_contrastive_loss_per_sample)
    
        return supervised_contrastive_loss

    def _extract_columns(self, x, z, cls_indices=None):
        """Helper function for extracting column vectors from LM outputs.
        """
        x_flat = x.view(-1)
        column_vectors = z.view((x_flat.shape[0], -1))

        if cls_indices is None:
            indices = [idx for idx, token_id in enumerate(x_flat) \
                if token_id == self.cls_token_id]
        else:
            indices = []
            seq_len = x.shape[-1]
            for rid in range(len(cls_indices)):
                indices += [idx + rid * seq_len for idx in cls_indices[rid]]

        return column_vectors[indices]


    def inference(self, x):
        """Apply the model on a serialized table.

        Args:
            x (LongTensor): a batch of serialized tables

        Returns:
            Tensor: the column vectors for all tables
        """
        # x = x.to(self.device)
        x = x.cuda()
        z = self.bert(x)[0]
        # z = self.projector(z) # optional
        return self._extract_columns(x, z)


    def forward(self, x_ori, x_aug, cls_indices, supervised_signals, mode="simclr", task="None"):
        """Apply the model for contrastive learning.

        Args:
            x_ori (LongTensor): the first views of a batch of tables
            x_aug (LongTensor): the second views of a batch of tables
            cls_indices (tuple of List): the cls_token alignment
            mode (str, optional): the name of the contrastive learning algorithm
            task (str, optional): the supervision signal, unsupervised if task == "None"

        Returns:
            Tensor: the loss or the column embeddings of x_ori, x_aug
        """
        if mode in ["simclr", "barlow_twins", "supcl", "supcon"]:
            # pre-training
            # encode
            batch_size = len(x_ori)
            x_ori = x_ori.cuda() # original, (batch_size, seq_len)
            x_aug = x_aug.cuda() # augment, (batch_size, seq_len)

            # encode each table (all columns)
            x = torch.cat((x_ori, x_aug)) # (2*batch_size, seq_len)
            z = self.bert(x)[0] # (2*batch_size, seq_len, hidden_size)

            # assert that x_ori and x_aug have the same number of columns
            z_ori = z[:batch_size] # (batch_size, seq_len, hidden_size)
            z_aug = z[batch_size:] # (batch_size, seq_len, hidden_size)

            if cls_indices is None:
                cls_ori = None
                cls_aug = None
            else:
                cls_ori, cls_aug = cls_indices
                
            z_ori_col = self._extract_columns(x_ori, z_ori, cls_ori) # (total_num_columns, hidden_size)
            z_aug_col = self._extract_columns(x_aug, z_aug, cls_aug) # (total_num_columns, hidden_size)
            assert z_ori_col.shape == z_aug_col.shape

            z_col = torch.cat((z_ori_col, z_aug_col))
            z_col = self.projector(z_col) # (2*total_num_columns, projector_size)

            if mode == "simclr":
                # simclr
                logits, labels = self.info_nce_loss(z_col, len(z_col) // 2, 2)
                loss = self.criterion(logits, labels)            
                return loss
            elif mode == "barlow_twins":
                # barlow twins
                z1 = z[:len(z) // 2]
                z2 = z[len(z) // 2:]
                # empirical cross-correlation matrix
                c = (self.bn(z1).T @ self.bn(z2)) / (len(z1))
                # use --scale-loss to multiply the loss by a constant factor
                on_diag = ((torch.diagonal(c) - 1) ** 2).sum() * self.hp.scale_loss
                off_diag = (off_diagonal(c) ** 2).sum() * self.hp.scale_loss
                loss = on_diag + self.hp.lambd * off_diag
                return loss
            elif mode in ["supcon", "supcon_ddp"]:
                if supervised_signals is not None:
                    loss = self.supcon_loss(z_col, len(z_col) // 2, 2, signals=supervised_signals)
                else:
                    # only return embedding, compute loss outside
                    return z_col
                return loss
                
        else:
            pass


class SupclLoss(nn.Module):
    """The class of loss function for computing the CL loss when using DDP and gather batches from all workers"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, signals=None):
        features = F.normalize(features, dim=1)
        # discard the main diagonal from both: labels and similarities matrix
        similarity_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        logits = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0].detach()
        if len(signals.shape) == 1:
            mask_similar_class = (signals.unsqueeze(1).repeat(1, signals.shape[0]) == signals).cuda()
        else:
            mask_similar_class = (torch.sum(torch.eq(signals.unsqueeze(1).repeat(1,signals.shape[0],1),
                                                     signals.unsqueeze(0).repeat(signals.shape[0],1,1)), dim=-1) == signals.shape[-1]).cuda()
            
        mask_anchor_out = (1 - torch.eye(logits.shape[0])).cuda()
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        
        # log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        exp_logits = torch.exp(logits) * mask_anchor_out
        log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = -torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

class BertMultiPairPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        #token_tensor = torch.index_select(hidden_states, 1,
        #                                  cls_indexes)
        # Apply
        #pooled_outputs = self.dense(token_tensor)
        # print(hidden_states.shape)
        hidden_states_first_cls = hidden_states[:, 0].unsqueeze(1).repeat(
            [1, hidden_states.shape[1], 1])
        pooled_outputs = self.dense(
            torch.cat([hidden_states_first_cls, hidden_states], 2))
        pooled_outputs = self.activation(pooled_outputs)
        # pooled_outputs = pooled_outputs.squeeze(0)
        # print(pooled_outputs.shape)
        return pooled_outputs

class BertForMultiOutputClassification(nn.Module):

    def __init__(self, hp, device='cuda', lm='roberta', col_pair='None'):
        
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        self.col_pair = col_pair
        hidden_size = 768

        # projector
        self.projector = nn.Linear(hidden_size, hp.projector)
        '''Require all models using the same CLS token'''
        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp['roberta']).cls_token_id
        self.num_labels = hp.num_labels
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.hp.num_labels)
    
    def load_from_CL_model(self, model):
        '''load from models pre-trained with contrastive learning'''
        self.bert = model.bert
        self.projector = model.projector
        self.cls_token_id = model.cls_token_id
    
    def forward(
        self,
        input_ids=None,
        get_enc=False
    ):
        # BertModelMultiOutput
        bert_output = self.bert(input_ids)
        # Note: returned tensor contains pooled_output of all tokens (to make the tensor size consistent)
        pooled_output = bert_output[0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if get_enc:
            outputs = (logits, pooled_output)
        else:
            outputs = logits
        return outputs  # (loss), logits, (hidden_states), (attentions)
    

class BertForMultiOutputClassificationColPopl(nn.Module):

    def __init__(self, hp, device='cuda', lm='roberta', col_pair='None', n_seed_cols=-1, cls_for_md=False):
        
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        self.col_pair = col_pair
        self.n_seed_cols = 3 if n_seed_cols == -1 else n_seed_cols
        self.cls_for_md = cls_for_md
        if self.cls_for_md:
            self.n_seed_cols += 1
        hidden_size = 768

        # projector
        self.projector = nn.Linear(hidden_size, hp.projector)

        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp['roberta']).cls_token_id
        self.num_labels = hp.num_labels
        if self.n_seed_cols > 1:
            self.dense = nn.Linear(hidden_size * self.n_seed_cols, hidden_size)
            self.activation = nn.Tanh()
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.hp.num_labels)

    def load_from_CL_model(self, model):
        self.bert = model.bert
        self.projector = model.projector
        self.cls_token_id = model.cls_token_id
    

    def forward(
        self,
        input_ids=None,
        cls_indexes=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):
        # print(295, input_ids.shape)
        # BertModelMultiOutput
        if "distilbert" in self.hp.__dict__['shortcut_name']:
            bert_output = self.bert(
                input_ids,
                #cls_indexes,
                attention_mask=attention_mask
            )
        else:
            bert_output = self.bert(
                input_ids,
                #cls_indexes,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        if self.n_seed_cols == 1:
            pooled_output = bert_output[0][:, 0]
        else:
            hidden_states = bert_output[0]
            cls_outputs = hidden_states[cls_indexes[:,0], cls_indexes[:, 1]].reshape(hidden_states.shape[0], self.n_seed_cols, 768)
            pooled_output = cls_outputs[:, 0]
            for j in range(1, self.n_seed_cols):
                pooled_output = torch.cat([pooled_output, cls_outputs[:, j]], dim=1)
            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
      
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
      

        outputs = (logits, )

        
        return outputs  # (loss), logits, (hidden_states), (attentions)

