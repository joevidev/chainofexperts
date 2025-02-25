from transformers import AutoConfig, AutoModel

config_path = "r1-tiny-hf"
config = AutoConfig.from_pretrained(config_path)

model = AutoModel.from_config(config)

print(model)