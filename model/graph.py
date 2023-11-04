import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
import os

from torch_geometric.nn import GCNConv, GATConv

# GRAPH = 'GCN'
GRAPH = "GRAPHORMER"


# GRAPH = 'GAT'


class SelfAttention(nn.Module):  # 自注意力机制的模块，用于处理输入的隐藏状态
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.self = BartAttention(config.hidden_size, config.num_attention_heads,
                                  config.attention_probs_dropout_prob)  # 计算自注意力
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 层归一化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 丢弃

    def forward(self, hidden_states,
                attention_mask=None, output_attentions=False, extra_attn=None):
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self(
            hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions,
            extra_attn=extra_attn,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states=None,
            past_key_value=None,
            attention_mask=None,
            output_attentions: bool = False,
            extra_attn=None,
            only_attn=False,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if extra_attn is not None:
            attn_weights += extra_attn

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        if only_attn:
            return attn_weights_reshaped

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class GraphLayer(nn.Module):
    def __init__(self, config, last=False):
        super(GraphLayer, self).__init__()
        self.config = config

        class _Actfn(nn.Module):  # 定义激活函数
            def __init__(self):
                super(_Actfn, self).__init__()
                if isinstance(config.hidden_act, str):  # 判断 config.hidden_act 的类型，来确定要使用的激活函数，字符串或者函数类型
                    self.intermediate_act_fn = ACT2FN[config.hidden_act]
                else:
                    self.intermediate_act_fn = config.hidden_act

            def forward(self, x):  # 将输入 x 应用于内部定义的激活函数 self.intermediate_act_fn，并返回激活后的结果
                return self.intermediate_act_fn(x)

        if GRAPH == 'GRAPHORMER':
            self.hir_attn = SelfAttention(config)
        elif GRAPH == 'GCN':
            self.hir_attn = GCNConv(config.hidden_size, config.hidden_size)
        elif GRAPH == 'GAT':
            self.hir_attn = GATConv(config.hidden_size, config.hidden_size, 1)

        self.last = last  # 是否为最后一层
        if last:
            self.cross_attn = BartAttention(config.hidden_size, 8, 0.1, True)  # 创建一个交叉注意力层，用于处理跨层注意力
            self.cross_layer_norm = nn.LayerNorm(config.hidden_size,
                                                 eps=config.layer_norm_eps)  # 创建一个层归一化层，用于规范化交叉注意力层的输出
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 创建一个线性层，用于最终的分类预测
        self.output_layer = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size),  #
                                          # 包含两个线性层和一个激活函数的序列。这些层的输入和输出维度都是 config.hidden_size。
                                          # 在输入层和输出层之间，应用了一个激活函数_Actfn()。
                                          _Actfn(),
                                          nn.Linear(config.intermediate_size, config.hidden_size),
                                          )
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 在前向传播过程中进行随机失活，以防止过拟合

    def forward(self, label_emb, extra_attn, self_attn_mask, inputs_embeds, cross_attn_mask):
        if GRAPH == 'GRAPHORMER':
            label_emb = self.hir_attn(label_emb,  # 通过 hir_attn 属性的调用，使用自注意力模块对 label_emb 进行自注意力计算
                                      attention_mask=self_attn_mask, extra_attn=extra_attn)[0]
            # label_emb = self.output_layer_norm(self.dropout(self.output_layer(label_emb)) + label_emb)
        elif GRAPH == 'GCN' or GRAPH == 'GAT':
            label_emb = self.hir_attn(label_emb.squeeze(0), edge_index=extra_attn)
        if self.last:
            label_emb = label_emb.expand(inputs_embeds.size(0), -1, -1)  # 将 label_emb 扩展为与 inputs_embeds 相同的大小
            label_emb = self.cross_attn(inputs_embeds, label_emb,
                                        attention_mask=cross_attn_mask.unsqueeze(1), output_attentions=True,
                                        only_attn=True)  # 通过 cross_attn 属性的调用，在 inputs_embeds 和 label_emb
            # 之间进行交叉注意力计算。注意力掩码 cross_attn_mask 用于控制注意力的计算范围。最后，返回交叉注意力计算结果 label_emb。
            return label_emb

        label_emb = self.output_layer_norm(self.dropout(self.output_layer(label_emb)) + label_emb)  # 如果不是最后一层，将通过
        # output_layer 属性对 label_emb 进行线性变换和激活函数处理，然后通过层归一化和随机失活进行规范化。最后返回处理后的 label_emb。
        if self.last:
            label_emb = self.dropout(self.classifier(label_emb))  # 返回预测结果 label_emb
        return label_emb


class GraphEncoder(nn.Module):
    def __init__(self, config, graph=False, layer=1, data_path=None, threshold=0.01, tau=1):
        super(GraphEncoder, self).__init__()
        self.config = config
        self.tau = tau
        self.label_dict = torch.load(os.path.join(data_path,
                                                  'bert_value_dict.pt'))  # 标签名称的字典。通过使用 torch.load 函数加载字典数据，并将标签名称解码为文本形式，存储在 self.label_dict 中
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # 加载预训练的 BERT tokenizer编码结果

        self.label_dict = {i: self.tokenizer.decode(v) for i, v in self.label_dict.items()}  # 并将标签名称进行分词和编码处理
        self.label_name = []
        for i in range(len(self.label_dict)):
            self.label_name.append(self.label_dict[i])  # 存储在 self.label_name 中
        self.label_name = self.tokenizer(self.label_name, padding='longest')['input_ids']
        # elf.label_name 转换为 torch.tensor 类型，并标记为不可训练的参数（requires_grad=False
        self.label_name = nn.Parameter(torch.tensor(self.label_name, dtype=torch.long), requires_grad=False)
        # 创建了多层的图编码器，每一层通过调用 GraphLayer 类进行构建。根据参数 layer 的值，决定最后一层是否为最后一层，从而设置 last 参数
        self.hir_layers = nn.ModuleList([GraphLayer(config, last=i == layer - 1) for i in range(layer)])
        self.label_num = len(self.label_name)
        self.graph = graph
        self.threshold = threshold

        if graph:
            label_hier = torch.load(os.path.join(data_path, 'slot.pt'))  # slot.pt 其中包含标签的层次信息
            path_dict = {}
            num_class = 0
            for s in label_hier:  # 对于每个标签 v，将其作为键，其所属的类别 s 作为值，添加到 path_dict 字典
                for v in label_hier[s]:
                    path_dict[v] = s
                    if num_class < v:  # 更新 num_class 的值为更大的类别值
                        num_class = v
            if GRAPH == 'GRAPHORMER':
                num_class += 1
                for i in range(num_class):
                    if i not in path_dict:  # 检查它是否存在于 path_dict 字典中。如果不存在，将其添加到 path_dict 字典中，并将其值设置为 i。
                        # 确保所有类别都在 path_dict 字典中有对应的条目
                        path_dict[i] = i
                self.inverse_label_list = {}  # 定义了一个空字典 inverse_label_list，用于后续将标签映射回类别

                def get_root(path_dict, n):  # 接受 path_dict 字典和一个标签 n
                    ret = []
                    while path_dict[n] != n:  # 返回从该标签到根节点的路径,通过从当前标签向上追溯其父节点，直到找到根节点（即父节点与自身相等）为止
                        # 并将路径保存在列表 ret 中。最终，返回路径列表 ret。
                        ret.append(n)
                        n = path_dict[n]
                    ret.append(n)
                    return ret

                for i in range(num_class):
                    # 每个类别的索引 i，与get_root(path_dict, i)+[-1]的结果进行合并，并将其作为键值对{i:get_root(path_dict,i)+[-1]},添加到inverse_label_list字典中
                    # 建立每个类别索引与从该类别到根节点的路径的映射关系,路径末尾的 -1 是为了表示根节点
                    self.inverse_label_list.update({i: get_root(path_dict, i) + [-1]})
                label_range = torch.arange(len(self.inverse_label_list))  # 创建一个范围为类别索引数量的张量，并将其赋值给 label_range
                self.label_id = label_range
                node_list = {}

                def get_distance(node1, node2):  # 计算两个类别索引之间的距离
                    p = 0
                    q = 0
                    node_list[(node1, node2)] = a = []  # 存储已经计算过的类别索引对之间的距离
                    node1 = self.inverse_label_list[node1]
                    node2 = self.inverse_label_list[node2]
                    while p < len(node1) and q < len(node2):
                        if node1[p] > node2[q]:
                            a.append(node1[p])
                            p += 1

                        elif node1[p] < node2[q]:
                            a.append(node2[q])
                            q += 1

                        else:
                            break
                    return p + q  # 类别索引对之间的距离

                # 将self.label_id转换为形状为(1, -1)的张量，使用repeat函数将其复制为形状为(self.label_id.size(0),self.label_id.size(0))的张量
                # 得到距离矩阵 self.distance_mat,阵用于存储所有类别索引之间的距离
                self.distance_mat = self.label_id.reshape(1, -1).repeat(self.label_id.size(0), 1)
                hier_mat_t = self.label_id.reshape(-1, 1).repeat(1, self.label_id.size(0))
                self.distance_mat.map_(hier_mat_t, get_distance)
                self.distance_mat = self.distance_mat.view(1, -1)
                self.edge_mat = torch.zeros(len(self.inverse_label_list), len(self.inverse_label_list), 15,
                                            dtype=torch.long)  # 全零张量用于存储边信息
                for i in range(len(self.inverse_label_list)):
                    for j in range(len(self.inverse_label_list)):
                        edge_list = node_list[(i, j)]
                        self.edge_mat[i, j, :len(edge_list)] = torch.tensor(edge_list) + 1
                self.edge_mat = self.edge_mat.view(-1, self.edge_mat.size(-1))
                # 将类别索引映射为对应的嵌入向量。它的输入维度为 len(self.inverse_label_list) + 1，输出维度为 config.hidden_size
                self.id_embedding = nn.Embedding(len(self.inverse_label_list) + 1, config.hidden_size,
                                                 len(self.inverse_label_list))
                self.distance_embedding = nn.Embedding(20, 1, 0)  # 将距离值映射为对应的嵌入向量
                self.edge_embedding = nn.Embedding(len(self.inverse_label_list) + 1, 1, 0)  # 将边关系值映射为对应的嵌入向量
                self.label_id = nn.Parameter(self.label_id, requires_grad=False)
                self.edge_mat = nn.Parameter(self.edge_mat, requires_grad=False)
                self.distance_mat = nn.Parameter(self.distance_mat, requires_grad=False)
            # 存储边缘关系的索引对。由path_dict中的键值对组成，每个键值对表示两个类别之间存在边缘关系。列表中的每个元素是一个包含两个索引的列表，被转置为形状为 (2, num_edges) 的张量
            self.edge_list = [[v, i] for v, i in path_dict.items()]
            self.edge_list += [[i, v] for v, i in path_dict.items()]
            self.edge_list = nn.Parameter(torch.tensor(self.edge_list).transpose(0, 1), requires_grad=False)

    def forward(self, inputs_embeds, attention_mask, labels, embeddings):
        label_mask = self.label_name != self.tokenizer.pad_token_id
        # full name
        label_emb = embeddings(self.label_name)  # 形状为 (batch_size, hidden_size)，其中 hidden_size 是嵌入向量的维度
        # label_emb 与 label_mask 进行元素级乘法，并在维度1上求和，然后除以 label_mask 在维度1上的求和，计算出每个标签的平均嵌入向量
        # 最终得到的 label_emb 的形状为 (1, hidden_size)，通过 unsqueeze(0) 在第0维上添加了一个维度，以与输入的 inputs_embeds 形状匹配
        label_emb = (label_emb * label_mask.unsqueeze(-1)).sum(dim=1) / label_mask.sum(dim=1).unsqueeze(-1)
        label_emb = label_emb.unsqueeze(0)
        # 创建了一个大小为 (1, label_emb.size(1)) 的全1张量 label_attn_mask，用于控制 label_emb 的自注意力机制
        label_attn_mask = torch.ones(1, label_emb.size(1), device=label_emb.device)

        extra_attn = None

        self_attn_mask = (label_attn_mask * 1.).t().mm(label_attn_mask * 1.).unsqueeze(0).unsqueeze(0)
        cross_attn_mask = (attention_mask * 1.).unsqueeze(-1).bmm(
            (label_attn_mask.unsqueeze(0) * 1.).repeat(attention_mask.size(0), 1, 1))
        expand_size = label_emb.size(-2) // self.label_name.size(0)  # 确定每个标签的嵌入向量在 label_emb 中的重复次数
        if self.graph:
            if GRAPH == 'GRAPHORMER':
                label_emb += self.id_embedding(self.label_id[:, None].expand(-1, expand_size)).view(1, -1,
                                                                                                    self.config.hidden_size)
                extra_attn = self.distance_embedding(self.distance_mat) + self.edge_embedding(self.edge_mat).sum(
                    dim=1) / (
                                     self.distance_mat.view(-1, 1) + 1e-8)
                extra_attn = extra_attn.view(self.label_num, 1, self.label_num, 1).expand(-1, expand_size, -1,
                                                                                          expand_size)
                extra_attn = extra_attn.reshape(self.label_num * expand_size, -1)
            elif GRAPH == 'GCN' or GRAPH == 'GAT':
                extra_attn = self.edge_list
        for hir_layer in self.hir_layers:
            label_emb = hir_layer(label_emb, extra_attn, self_attn_mask, inputs_embeds, cross_attn_mask)

        token_probs = label_emb.mean(dim=1).view(attention_mask.size(0), attention_mask.size(1),
                                                 self.label_name.size(0),
                                                 )

        # sum
        contrast_mask = (F.gumbel_softmax(token_probs, hard=False, dim=-1, tau=self.tau) * labels.unsqueeze(1)).sum(
            -1)

        temp = self.threshold
        _mask = contrast_mask > temp
        contrast_mask = contrast_mask + (1 - contrast_mask).detach()
        contrast_mask = contrast_mask * _mask

        return contrast_mask
