import torch
import torch.nn as nn


class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,):
        super().__init__()

        self.length = length  # 5
        self.prompt_pool = prompt_pool  # True
        self.embedding_key = embedding_key  # 'cls'
        self.prompt_init = prompt_init  # 'uniform'
        self.prompt_key = prompt_key  # False
        self.pool_size = pool_size  # 10
        self.top_k = top_k  # 1
        self.batchwise_prompt = batchwise_prompt  # False
        self.num_layers = num_layers  # 5
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt  # True
        self.num_heads = num_heads  # 12
        self.same_key_value = same_key_value  # False

        if self.prompt_pool:  # True
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:  # True
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length,   # 5, 2, 10, 5
                                        self.num_heads, embed_dim // self.num_heads)  # 12, 64
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads. 
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length, embed_dim)  # TODO fix self.num_layers = 1
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                    
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])  # 在layer和所有的prompt方向求均值
            self.prompt_key = prompt_mean   # 2,5,12,64
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, prompt_idx=None, prompt_weight=None, prompt_momentum=0):  # 训练时prompt_mask根据task id得到, prompt_idx根据tii推理的cls得到
        assert prompt_mask is not None or prompt_idx is not None or prompt_weight is not None
        assert self.prompt_pool, "In HiDe-Prompt, 'prompt_pool' must be set to True"
        out = dict()
        if self.prompt_pool:
            idx = prompt_idx  # 根据第一轮的cls选择expert prompt index

            if self.batchwise_prompt and prompt_idx is not None:
                prompt_id, id_counts = torch.unique(prompt_idx, return_counts=True, sorted=True)
                
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(prompt_idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                major_prompt_id = prompt_id[major_idx] # top_k
                # expand to batch
                idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous()  # B, top_k
            
            if prompt_mask is not None:  # 训练时有prompt_mask；所以是根据task id，而不是第一轮推理得到的prompt id，选择prompt
                idx = prompt_mask  # B, top_k;  根据task id选择expert prompt index；训练时根据task id指定prompt
            if idx is not None:
                out['prompt_idx'] = idx
            if self.use_prefix_tune_for_e_prompt:
                if prompt_weight is not None:  # 固定权重加权
                    batched_prompt_raw = torch.einsum("bp,ndplhe->ndblhe", prompt_weight, self.prompt) # num_layers, 2, B, top_k, length, C
                    batched_prompt_raw = batched_prompt_raw.unsqueeze(3)
                    num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                    # print(top_k)
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                    )
                elif prompt_momentum > 0 and prompt_mask is not None:  # 根据task id选中的prompt和当前patch中第一个样本的首个prompt加权
                    with torch.no_grad():  # 选择过程不要记录梯度; prompt本身的维度[5,2, 10, 5,12,64]; 选择该batch中第一个样本(的第一个)选中的prompt作为基础进行repeat
                        batched_prompt_momentum = self.prompt[:, :, 0:idx[0][0]].detach().clone().mean(2, keepdim=True).unsqueeze(2).repeat(1,1,idx.shape[0],1,1,1,1)  # [5,2, 24,1, 5,12,64]
                    batched_prompt_raw = (1-prompt_momentum) * self.prompt[:, :, idx] + prompt_momentum * batched_prompt_momentum
                    num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                    )

                else:  # 根据expert prompt index选择expert prompt
                    batched_prompt_raw = self.prompt[:, :, idx]  # num_layers, B, top_k, length, C;  self.prompt=[5,2,10,5,12,64]; betched_prompt_raw=[5,2,24,1,5,12,64]
                    num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape  # [5,2,24,1,5,12,64]
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                    )  # [5, 24, 2, 5, 12, 64]
            else:
                if prompt_weight is not None:
                    batched_prompt_raw = torch.einsum("bp,npld->nbpld", prompt_weight, self.prompt)
                    num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, top_k * length, embed_dim
                    )
                else:
                    batched_prompt_raw = self.prompt[:, idx]
                    num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, top_k * length, embed_dim
                    )
        
        out['batched_prompt'] = batched_prompt

        return out
