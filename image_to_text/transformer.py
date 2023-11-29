from PIL import Image

import torch
import torch.nn.functional as F

from torch import Tensor, nn
from pytorch_model_summary import summary
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size=224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )  # this breaks down the image in s1xs2 patches, and then flat them

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)  # prepending the cls token
        x += self.positions.unsqueeze(0).repeat(x.size(0), 1, 1)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)  # queries, keys and values matrix
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)

        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)  # sum over the third axis
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, L: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, L * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(L * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, L=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs): # initially 12
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ViTEncoder(nn.Sequential):
    def __init__(self,
                in_channels: int,
                patch_size: int,
                emb_size: int,
                img_size: int,
                depth: int,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
        )


class GPT2Decoder(nn.Module):
    def __init__(self, max_length, temperature):
        super(GPT2Decoder, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side="left")
        self.vocab_size = self.tokenizer.vocab_size
        self.config = GPT2Config.from_pretrained('gpt2', add_cross_attention=True)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', config=self.config)
        self.max_length = max_length
        self.temperature = temperature

    def forward(self, encoder_output, caption):
        logits = self.model(input_ids=caption, encoder_hidden_states=encoder_output).logits
        logits = torch.nn.functional.softmax(logits, dim=-1)
        loss = None

        if caption is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.vocab_size), caption.reshape(-1))

        return {"logits": logits, "loss": loss}

    def generate(self, encoder_output, max_new_tokens):
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).cuda()

        for i in range(max_new_tokens):
            logits = self.model(input_ids, encoder_hidden_states=encoder_output).logits
            logits = logits[:, -1, :] / self.temperature

            next_token_probs = torch.nn.functional.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(next_token_probs, 1).squeeze(-1)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)

        generated_text = self.tokenizer.decode(input_ids[:, 1:].squeeze(), skip_special_tokens=True)
        return generated_text

class ImageCaptioningModel(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 depth: int = 12,
                 max_length: int = 50,
                 temperature: float = 1.0,
                 device: str = "cuda",
                 **kwargs):
        super(ImageCaptioningModel, self).__init__()
        self.device = device
        self.vit_encoder = ViTEncoder(in_channels, patch_size, emb_size, img_size, depth, **kwargs)
        self.torch_decoder = GPT2Decoder(max_length, temperature)

    def forward(self, pixel_values, labels):
        image_features = self.vit_encoder(pixel_values)
        generated_captions = self.torch_decoder(image_features, labels)

        return generated_captions

    def generate(self, pixel_values, max_new_tokens):
        image_features = self.vit_encoder(pixel_values)
        return self.torch_decoder.generate(image_features, max_new_tokens=max_new_tokens)
