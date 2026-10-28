import torch
import torch.nn as nn
import torch.nn.functional as F

from flask import Flask, jsonify, send_file
import io
import base64
from PIL import Image


# =========================================================
# DEVICE
# =========================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# =========================================================
# MODEL CLASSES
# Same architecture used in your Colab
# =========================================================

class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(
            torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8
        )


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=256, w_dim=256, n_layers=6):
        super().__init__()

        layers = [PixelNorm()]
        in_dim = z_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = w_dim

        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)


class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(
            channels,
            affine=False
        )

        self.style_scale = nn.Linear(w_dim, channels)
        self.style_bias = nn.Linear(w_dim, channels)

        nn.init.ones_(self.style_scale.bias)
        nn.init.zeros_(self.style_bias.bias)

    def forward(self, x, w):
        x = self.instance_norm(x)

        scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)

        return scale * x + bias


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

    def forward(self, x, noise=None):

        if noise is None:
            noise = torch.randn(
                x.size(0),
                1,
                x.size(2),
                x.size(3),
                device=x.device
            )

        return x + self.weight * noise


class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim, upsample=True):
        super().__init__()

        self.upsample = upsample

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            3,
            padding=1
        )

        self.noise1 = NoiseInjection(out_ch)
        self.adain1 = AdaIN(out_ch, w_dim)

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            3,
            padding=1
        )

        self.noise2 = NoiseInjection(out_ch)
        self.adain2 = AdaIN(out_ch, w_dim)

        self.act = nn.LeakyReLU(
            0.2,
            inplace=True
        )

    def forward(self, x, w):

        if self.upsample:
            x = F.interpolate(
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )

        x = self.act(self.conv1(x))
        x = self.noise1(x)
        x = self.adain1(x, w)

        x = self.act(self.conv2(x))
        x = self.noise2(x)
        x = self.adain2(x, w)

        return x


class Generator(nn.Module):

    def __init__(
        self,
        w_dim=256,
        base_channels=256,
        img_channels=3
    ):
        super().__init__()

        self.channels = {
            4: 256,
            8: 256,
            16: 128,
            32: 64,
            64: 32
        }

        self.resolutions = [
            4,
            8,
            16,
            32,
            64
        ]

        self.starting_constant = nn.Parameter(
            torch.randn(
                1,
                self.channels[4],
                4,
                4
            )
        )

        self.initial_noise = NoiseInjection(
            self.channels[4]
        )

        self.initial_adain1 = AdaIN(
            self.channels[4],
            w_dim
        )

        self.initial_conv = nn.Conv2d(
            self.channels[4],
            self.channels[4],
            3,
            padding=1
        )

        self.initial_noise2 = NoiseInjection(
            self.channels[4]
        )

        self.initial_adain2 = AdaIN(
            self.channels[4],
            w_dim
        )

        self.act = nn.LeakyReLU(
            0.2,
            inplace=True
        )

        self.blocks = nn.ModuleList()

        for i in range(1, len(self.resolutions)):

            in_ch = self.channels[
                self.resolutions[i - 1]
            ]

            out_ch = self.channels[
                self.resolutions[i]
            ]

            self.blocks.append(
                SynthesisBlock(
                    in_ch,
                    out_ch,
                    w_dim,
                    upsample=True
                )
            )

        self.to_rgb = nn.Conv2d(
            self.channels[64],
            img_channels,
            kernel_size=1
        )

        self.num_style_inputs = 1 + len(self.blocks)

    def forward(self, ws):

        if isinstance(ws, torch.Tensor):
            ws = [ws] * self.num_style_inputs

        batch = ws[0].shape[0]

        x = self.starting_constant.expand(
            batch,
            -1,
            -1,
            -1
        )

        x = self.initial_noise(x)
        x = self.initial_adain1(x, ws[0])

        x = self.act(
            self.initial_conv(x)
        )

        x = self.initial_noise2(x)
        x = self.initial_adain2(x, ws[0])

        for i, block in enumerate(self.blocks):

            x = block(
                x,
                ws[i + 1]
            )

        return torch.tanh(
            self.to_rgb(x)
        )


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

Z_DIM = 256
W_DIM = 256

mapping_network = MappingNetwork(
    Z_DIM,
    W_DIM,
    n_layers=6
).to(device)

generator = Generator(
    w_dim=W_DIM,
    base_channels=256
).to(device)


MODEL_PATH = "model/stylegan_trained.pth"

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

mapping_network.load_state_dict(
    checkpoint["mapping_network"]
)

generator.load_state_dict(
    checkpoint["generator"]
)

mapping_network.eval()
generator.eval()

print("✅ StyleGAN model loaded successfully!")


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


@app.route("/")
def home():

    return send_file("index.html")


@app.route("/generate")
def generate():

    with torch.no_grad():

        z = torch.randn(
            1,
            Z_DIM,
            device=device
        )

        w = mapping_network(z)

        image = generator(w)

    # Convert [-1, 1] → [0, 1]
    image = (image + 1) / 2

    image = image.clamp(0, 1)

    # CHW → HWC
    image = image[0].permute(
        1,
        2,
        0
    )

    image = (
        image.cpu().numpy() * 255
    ).astype("uint8")

    pil_image = Image.fromarray(
        image
    )

    # Convert image to base64
    buffer = io.BytesIO()

    pil_image.save(
        buffer,
        format="PNG"
    )

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return jsonify({
        "image": image_base64
    })


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )