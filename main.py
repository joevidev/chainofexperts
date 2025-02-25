from transformers import AutoConfig, AutoModel

config_path = "config/r1-tiny-hf"
config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

model = AutoModel.from_config(config, trust_remote_code=True)

print(model)
breakpoint()