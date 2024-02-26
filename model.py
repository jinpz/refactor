import torch
from torch_geometric.nn import GraphConv, GINConv, SAGEConv
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import math
from torch.nn import Linear, Sequential, ReLU


class Counter:
    def __init__(self, name):
        self.acc = []
        self.name = name


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class BidirectionalRNN_old(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, rnn_type, concat_layer, bidirectional, last_hidden_only):
        super(BidirectionalRNN_old, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.concat_layers = concat_layer
        self.rnn_list = nn.ModuleList()
        if bidirectional == 1:
            self.bidirectional = True
        else:
            self.bidirectional = False
        self.last_hidden_only = last_hidden_only
        for i in range(num_layers):
            input_size = input_size if i == 0 else hidden_size * 2
            self.rnn_list.append(rnn_type(input_size, hidden_size, num_layers=1, bidirectional=self.bidirectional))

    def forward(self, x, x_mask):
        lengths = x_mask.data.eq(0).sum(1)
        _, index_sort = torch.sort(lengths, dim=0, descending=True)
        _, index_unsort = torch.sort(index_sort, dim=0, descending=False)
        lengths = list(lengths[index_sort])
        x = x.index_select(dim=0, index=index_sort)
        x = x.transpose(0, 1)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0.0:
                dropout_input = F.dropout(rnn_input.data, p=self.dropout_rate, training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input, rnn_input.batch_sizes)
            outputs.append(self.rnn_list[i](rnn_input)[0])
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]
        if self.concat_layers:
            output = torch.cat(outputs[1:], dim=2)
        else:
            output = outputs[-1]
        output = output.transpose(0, 1)
        output = output.index_select(dim=0, index=index_unsort)
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0), x_mask.size(1) - output.size(1), output.size(2)).type(
                output.data.type())
            output = torch.cat([output, padding], 1)
        if self.last_hidden_only:
            indices = x_mask.shape[1] - x_mask.sum(dim=1) - 1
            output = output[torch.arange(output.shape[0]), indices, :]
        return output


class BidirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, bidirectional, last_hidden_only):
        super(BidirectionalRNN, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        if bidirectional == 1:
            self.bidirectional = True
        else:
            self.bidirectional = False
        self.last_hidden_only = last_hidden_only
        self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=self.bidirectional)

    def forward(self, x, x_mask):
        output, (h_n, c_n) = self.model.forward(x)
        if self.last_hidden_only:
            indices = (x_mask.sum(dim=1) - 1).long()
            output = output[torch.arange(output.shape[0]), indices, :]
        return output


class BaseClassifier(pl.LightningModule):
    def __init__(self, learning_rate, lam, average, scaling_factor, ams_grad, eps):
        super().__init__()
        self.train_counter = Counter('train')
        self.val_counter = Counter('val')
        self.lr = learning_rate
        self.lam = lam
        self.average = average  # either 'node' or 'proof'. Node will do a simple average on all nodes. Proof will do
        # an average across nodes for each proof and then average across all proofs.
        self.scaling_factor = scaling_factor
        self.ams_grad = bool(ams_grad)
        self.eps = eps

    def forward(self, data):
        pass

    def on_epoch_start(self):
        print('modifying trainer length')
        epoch_train_length = len(self.trainer.train_dataloader.batch_sampler)
        self.trainer.num_training_batches = epoch_train_length
        self.trainer.val_check_batch = epoch_train_length
        print('epoch started')

    @staticmethod
    def get_proof_level_acc(node_correctness, batch_batch):
        proof_level_acc = torch.zeros((batch_batch[-1].item() + 1, )).to(batch_batch.device)
        for i in range(batch_batch[-1].item() + 1):
            current_correctness = node_correctness[batch_batch == i]
            count = current_correctness.long().sum().item()
            if count != current_correctness.shape[0]:
                proof_level_acc[i] = 0
            else:
                proof_level_acc[i] = 1
        return proof_level_acc

    @staticmethod
    def get_proof_level_loss(node_loss, batch_batch):
        coefficients = torch.zeros_like(node_loss)
        for i in range(batch_batch[-1].item() + 1):
            current_indices = batch_batch == i
            length = current_indices.float().sum()
            coefficients[current_indices] = 1 / length
        coefficients = coefficients / (batch_batch[-1].item() + 1)
        total_loss = (node_loss * coefficients).sum()
        return total_loss

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        weights = torch.ones_like(batch.y) / self.scaling_factor + (1.0 - 1.0/self.scaling_factor)*batch.y
        raw_loss = F.binary_cross_entropy(y_hat, batch.y, weight=weights, reduction='none')
        proof_level_loss = self.get_proof_level_loss(raw_loss, batch.batch)
        raw_loss_mean = raw_loss.mean()
        node_correctness = (batch.y == y_hat.round())
        node_level_acc = node_correctness.float().mean()
        proof_level_acc = self.get_proof_level_acc(node_correctness, batch.batch).mean()
        if self.average == 'node':
            loss = raw_loss_mean
        elif self.average == 'proof':
            loss = proof_level_loss
        else:
            raise NotImplementedError
        result = pl.TrainResult(loss)
        result.log('train_loss_node', raw_loss_mean, on_step=False, on_epoch=True, prog_bar=True)
        result.log('train_loss_proof', proof_level_loss, on_step=False, on_epoch=True, prog_bar=True)
        result.log('train_node_acc', node_level_acc, on_step=False, on_epoch=True, prog_bar=True)
        result.log('train_proof_acc', proof_level_acc, on_step=False, on_epoch=True, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        weights = torch.ones_like(batch.y) / self.scaling_factor + (1.0 - 1.0/self.scaling_factor)*batch.y
        raw_loss = F.binary_cross_entropy(y_hat, batch.y, weight=weights, reduction='none')
        proof_level_loss = self.get_proof_level_loss(raw_loss, batch.batch)
        raw_loss_mean = raw_loss.mean()
        node_correctness = (batch.y == y_hat.round())
        node_level_acc = node_correctness.float().mean()
        proof_level_acc = self.get_proof_level_acc(node_correctness, batch.batch).mean()
        if self.average == 'node':
            loss = raw_loss_mean
        elif self.average == 'proof':
            loss = proof_level_loss
        else:
            raise NotImplementedError
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss_node', raw_loss_mean)
        result.log('val_loss_proof', proof_level_loss)
        result.log('val_node_acc', node_level_acc, prog_bar=True)
        result.log('val_proof_acc', proof_level_acc, prog_bar=True)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.lam, amsgrad=self.ams_grad, eps=self.eps)


class ParallelPrediction(BaseClassifier):
    def __init__(self, num_words, embed_dim, max_length, learning_rate, pos_embed, mlp_after_embed, embed_agg, lam, average, num_layers, hidden, scaling_factor, ams_grad, eps, **unused):
        super().__init__(learning_rate, lam, average, scaling_factor, ams_grad, eps)
        self.save_hyperparameters()
        self.pos_embed = pos_embed
        self.mlp_after_embed = mlp_after_embed
        self.item_embedding = torch.nn.Embedding(num_embeddings=num_words + 1, embedding_dim=embed_dim)
        if self.pos_embed:
            self.pos_encoding = PositionalEncoding(embed_dim, dropout=0, max_len=max_length)
        self.conv1 = GraphConv(embed_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GraphConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(hidden, hidden)
        self.lin2 = torch.nn.Linear(hidden, 1)
        if self.mlp_after_embed:
            self.lin3 = torch.nn.Linear(embed_dim, hidden)
            self.lin4 = torch.nn.Linear(hidden, embed_dim)

        self.max_length = max_length
        self.num_words = num_words
        self.embed_dim = embed_dim
        self.embed_agg = embed_agg

    def forward(self, data):
        x, edge_index, batch = data.node_features, data.edge_index, data.batch
        x = [node_feature for proof in x for node_feature in proof]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.num_words)
        word_indices = x.long()
        word_feature = self.item_embedding(word_indices)
        if self.pos_embed:
            word_feature = self.pos_encoding(word_feature)
        word_mask = (word_indices != self.num_words).float().unsqueeze(2).repeat(1, 1, self.embed_dim)
        if self.mlp_after_embed:
            word_feature = self.lin3(word_feature)
            word_feature = self.lin4(word_feature)
        if self.embed_agg == 'mean':
            features = (word_feature * word_mask).mean(1)
        elif self.embed_agg == 'sum':
            features = (word_feature*word_mask).sum(1)
        else:
            raise NotImplementedError

        x = F.relu(self.conv1(features, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = x.squeeze()

        return x


class ParallelPrediction_LSTM(BaseClassifier):
    # obsolete
    def __init__(self, num_words, embed_dim, max_length, learning_rate, mlp_after_embed, embed_agg, lam, lstm_bidirectional, last_hidden_only, average, scaling_factor, **unused):
        super().__init__(learning_rate, lam, average, scaling_factor)
        self.save_hyperparameters()
        self.mlp_after_embed = mlp_after_embed
        self.item_embedding = torch.nn.Embedding(num_embeddings=num_words + 1, embedding_dim=embed_dim)
        self.rnn = BidirectionalRNN(embed_dim, embed_dim // 2, 1, 0, nn.LSTM, False, lstm_bidirectional, last_hidden_only)
        self.conv1 = GraphConv(embed_dim, 128)
        self.conv2 = GraphConv(128, 128)
        self.conv3 = GraphConv(128, 128)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 1)
        if self.mlp_after_embed:
            if lstm_bidirectional:
                self.lin3 = torch.nn.Linear(embed_dim, 64)
            else:
                self.lin3 = torch.nn.Linear(embed_dim // 2, 64)
            self.lin4 = torch.nn.Linear(64, embed_dim)

        self.max_length = max_length
        self.num_words = num_words
        self.embed_dim = embed_dim
        self.embed_agg = embed_agg
        self.last_hidden_only = last_hidden_only

    def forward(self, data):
        x, edge_index, batch = data.node_features, data.edge_index, data.batch
        x = [node_feature for proof in x for node_feature in proof]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.num_words)
        word_indices = x.long()
        word_feature = self.item_embedding(word_indices)
        word_mask = (word_indices == self.num_words)
        word_feature = self.rnn(word_feature, word_mask)
        if self.mlp_after_embed:
            word_feature = self.lin3(word_feature)
            word_feature = self.lin4(word_feature)
        if not self.last_hidden_only:
            word_mask = (word_indices != self.num_words).float().unsqueeze(2).repeat(1, 1, self.embed_dim)
            if self.embed_agg == 'mean':
                features = (word_feature * word_mask).mean(1)
            elif self.embed_agg == 'sum':
                features = (word_feature*word_mask).sum(1)
            else:
                raise NotImplementedError
        else:
            features = word_feature

        x = F.relu(self.conv1(features, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = x.squeeze()

        return x


class GIN(BaseClassifier):
    def __init__(self, num_words, embed_dim, max_length, learning_rate, mlp_after_embed, embed_agg, lam, average, num_layers, hidden, scaling_factor, ams_grad, eps, **unused):
        super(GIN, self).__init__(learning_rate, lam, average, scaling_factor, ams_grad, eps)
        # use hidden size 64 for now
        self.save_hyperparameters()
        self.mlp_after_embed = mlp_after_embed
        self.item_embedding = torch.nn.Embedding(num_embeddings=num_words + 1, embedding_dim=embed_dim)
        self.conv1 = GINConv(
            Sequential(
                Linear(embed_dim, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                    ), train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 1)
        if self.mlp_after_embed:
            self.lin3 = torch.nn.Linear(embed_dim, hidden)
            self.lin4 = torch.nn.Linear(hidden, embed_dim)

        self.max_length = max_length
        self.num_words = num_words
        self.embed_dim = embed_dim
        self.embed_agg = embed_agg

    def forward(self, data):
        x, edge_index, batch = data.node_features, data.edge_index, data.batch
        x = [node_feature for proof in x for node_feature in proof]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.num_words)
        word_indices = x.long()
        word_feature = self.item_embedding(word_indices)
        word_mask = (word_indices != self.num_words).float().unsqueeze(2).repeat(1, 1, self.embed_dim)
        if self.mlp_after_embed:
            word_feature = self.lin3(word_feature)
            word_feature = self.lin4(word_feature)
        if self.embed_agg == 'mean':
            features = (word_feature * word_mask).mean(1)
        elif self.embed_agg == 'sum':
            features = (word_feature*word_mask).sum(1)
        else:
            raise NotImplementedError

        x = self.conv1(features, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = x.squeeze()
        return x


class GraphSAGE(BaseClassifier):
    def __init__(self, num_words, embed_dim, max_length, learning_rate, mlp_after_embed, embed_agg, lam, average, num_layers, hidden, scaling_factor, ams_grad, eps, **unused):
        super(GraphSAGE, self).__init__(learning_rate, lam, average, scaling_factor, ams_grad, eps)
        self.save_hyperparameters()
        self.mlp_after_embed = mlp_after_embed
        self.item_embedding = torch.nn.Embedding(num_embeddings=num_words + 1, embedding_dim=embed_dim)
        self.conv1 = SAGEConv(embed_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 1)
        if self.mlp_after_embed:
            self.lin3 = torch.nn.Linear(embed_dim, hidden)
            self.lin4 = torch.nn.Linear(hidden, embed_dim)

        self.max_length = max_length
        self.num_words = num_words
        self.embed_dim = embed_dim
        self.embed_agg = embed_agg

    def forward(self, data):
        x, edge_index, batch = data.node_features, data.edge_index, data.batch
        x = [node_feature for proof in x for node_feature in proof]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.num_words)
        word_indices = x.long()
        word_feature = self.item_embedding(word_indices)
        word_mask = (word_indices != self.num_words).float().unsqueeze(2).repeat(1, 1, self.embed_dim)
        if self.mlp_after_embed:
            word_feature = self.lin3(word_feature)
            word_feature = self.lin4(word_feature)
        if self.embed_agg == 'mean':
            features = (word_feature * word_mask).mean(1)
        elif self.embed_agg == 'sum':
            features = (word_feature*word_mask).sum(1)
        else:
            raise NotImplementedError

        x = F.relu(self.conv1(features, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = x.squeeze()
        return x


class GraphSAGE_LSTM(BaseClassifier):
    def __init__(self, num_words, embed_dim, max_length, learning_rate, mlp_after_embed, embed_agg, lam, average, num_layers, hidden, lstm_num_layers, lstm_bidirectional, lstm_dropout, last_hidden_only, scaling_factor, ams_grad, eps, **unused):
        super(GraphSAGE_LSTM, self).__init__(learning_rate, lam, average, scaling_factor, ams_grad, eps)
        self.save_hyperparameters()
        self.mlp_after_embed = mlp_after_embed
        self.item_embedding = torch.nn.Embedding(num_embeddings=num_words + 1, embedding_dim=embed_dim)
        self.conv1 = SAGEConv(embed_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 1)
        if self.mlp_after_embed:
            self.lin3 = torch.nn.Linear(embed_dim, hidden)
            self.lin4 = torch.nn.Linear(hidden, embed_dim)
        if lstm_bidirectional:
            self.lstm = BidirectionalRNN(embed_dim, embed_dim // 2, lstm_num_layers, lstm_dropout, lstm_bidirectional, last_hidden_only)
        else:
            self.lstm = BidirectionalRNN(embed_dim, embed_dim, lstm_num_layers, lstm_dropout, lstm_bidirectional, last_hidden_only)
        self.max_length = max_length
        self.num_words = num_words
        self.embed_dim = embed_dim
        self.embed_agg = embed_agg
        self.last_hidden_only = last_hidden_only

    def forward(self, data):
        x, edge_index, batch = data.node_features, data.edge_index, data.batch
        x = [node_feature for proof in x for node_feature in proof]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.num_words)
        word_indices = x.long()
        word_feature = self.item_embedding(word_indices)
        word_mask = (word_indices != self.num_words).float().unsqueeze(2).repeat(1, 1, self.embed_dim)
        word_feature = self.lstm(word_feature, word_mask[:, :, 0])
        if self.mlp_after_embed:
            word_feature = self.lin3(word_feature)
            word_feature = self.lin4(word_feature)
        if not self.last_hidden_only:
            if self.embed_agg == 'mean':
                features = (word_feature * word_mask).mean(1)
            elif self.embed_agg == 'sum':
                features = (word_feature * word_mask).sum(1)
            else:
                raise NotImplementedError
        else:
            features = word_feature

        x = F.relu(self.conv1(features, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = x.squeeze()
        return x
